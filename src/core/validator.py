from typing import Dict, List, Any, Optional, Union, Set, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import re
import json
import logging
from datetime import datetime, date, time
from pathlib import Path
import statistics
from collections import defaultdict, Counter
import uuid

try:
    from pydantic import BaseModel, Field, validator, ValidationError, create_model
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    raise ImportError("Pydantic is required for validation. Install with: pip install pydantic")

try:
    from cerberus import Validator
    CERBERUS_AVAILABLE = True
except ImportError:
    CERBERUS_AVAILABLE = False
    raise ImportError("Cerberus is required for validation. Install with: pip install cerberus")

from src.core.schema_analyzer import SchemaAnalysisResult, TableAnalysis, ColumnAnalysis, DataCategory, ProviderType


class ValidationLevel(Enum):
    STRICT = "strict"
    STANDARD = "standard"
    LENIENT = "lenient"


class ValidationResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    table_name: str
    column_name: Optional[str]
    row_index: Optional[int]
    issue_type: str
    severity: ValidationResult
    message: str
    expected_value: Any = None
    actual_value: Any = None
    suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    total_records: int
    total_issues: int
    issues_by_severity: Dict[ValidationResult, int]
    issues_by_table: Dict[str, int]
    issues_by_type: Dict[str, int]
    detailed_issues: List[ValidationIssue]
    quality_score: float
    validation_timestamp: datetime
    validation_duration: float
    quality_metrics: Dict[str, Any]


class BusinessRuleValidator:
    
    @staticmethod
    def validate_email_domain(value: str, allowed_domains: List[str] = None) -> bool:
        if not value or '@' not in value:
            return False
        domain = value.split('@')[1].lower()
        if allowed_domains:
            return domain in [d.lower() for d in allowed_domains]
        return True
    
    @staticmethod
    def validate_phone_format(value: str, country_code: str = None) -> bool:
        if not value:
            return False
        clean_phone = re.sub(r'[\s\-\(\)]', '', value)
        if country_code == 'TN':
            return re.match(r'^\+?216\d{8}$', clean_phone) is not None
        elif country_code == 'US':
            return re.match(r'^\+?1?\d{10}$', clean_phone) is not None
        else:
            return re.match(r'^\+?\d{8,15}$', clean_phone) is not None
    
    @staticmethod
    def validate_postal_code(value: str, country_code: str = None) -> bool:
        if not value:
            return False
        if country_code == 'TN':
            return re.match(r'^\d{4}$', value) is not None
        elif country_code == 'US':
            return re.match(r'^\d{5}(-\d{4})?$', value) is not None
        elif country_code == 'FR':
            return re.match(r'^\d{5}$', value) is not None
        else:
            return re.match(r'^[A-Z0-9\s\-]{3,10}$', value.upper()) is not None
    
    @staticmethod
    def validate_credit_card(value: str) -> bool:
        if not value:
            return False
        clean_card = re.sub(r'\D', '', value)
        if len(clean_card) < 13 or len(clean_card) > 19:
            return False
        
        def luhn_check(card_num):
            def digits_of(number):
                return [int(d) for d in str(number)]
            
            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d*2))
            return checksum % 10 == 0
        
        return luhn_check(clean_card)
    
    @staticmethod
    def validate_date_consistency(start_date: Any, end_date: Any) -> bool:
        if not start_date or not end_date:
            return True
        try:
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            return start_date <= end_date
        except:
            return False


class CerberusSchemaBuilder:
    
    def __init__(self, business_rules: BusinessRuleValidator = None):
        self.business_rules = business_rules or BusinessRuleValidator()
        self.custom_validators = {}
        self._register_custom_validators()
    
    def _register_custom_validators(self):
        self.custom_validators = {
            'email_domain': self.business_rules.validate_email_domain,
            'phone_format': self.business_rules.validate_phone_format,
            'postal_code': self.business_rules.validate_postal_code,
            'credit_card': self.business_rules.validate_credit_card,
            'date_consistency': self.business_rules.validate_date_consistency,
        }
    
    def _get_cerberus_type(self, column_analysis: ColumnAnalysis) -> str:
        """Map Python types to Cerberus types"""
        type_mapping = {
            'str': 'string',
            'int': 'integer',
            'float': 'float',
            'bool': 'boolean',
            'datetime': 'datetime',
            'date': 'date',
            'time': 'time',
            'uuid': 'string',
            'decimal': 'float',
            'text': 'string',
            'json': 'dict',
            'list': 'list'
        }
        return type_mapping.get(column_analysis.python_type, 'string')
    
    def _build_column_schema(self, column_analysis: ColumnAnalysis, locale: str = 'en_US') -> Dict[str, Any]:
        """Build validation schema for a single column"""
        schema = {
            'type': self._get_cerberus_type(column_analysis),
            'required': not column_analysis.nullable,
            'nullable': column_analysis.nullable
        }
        
        if column_analysis.max_length:
            schema['maxlength'] = column_analysis.max_length
        
        
        
        if column_analysis.pattern_detected:
            patterns = {
                'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                'phone': r'^\+?[\d\s\-\(\)]+$',
                'url': r'^https?://[^\s/$.?#].[^\s]*$',
                'postal_code': r'^[A-Z0-9\s\-]{3,10}$',
                'credit_card': r'^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$',
                'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            }
            if column_analysis.pattern_detected in patterns:
                schema['regex'] = patterns[column_analysis.pattern_detected]
        
        if column_analysis.data_category == DataCategory.FINANCIAL:
            if 'credit_card' in column_analysis.name.lower():
                schema['check_with'] = 'credit_card'
            elif column_analysis.python_type in ['float', 'int']:
                schema['min'] = 0
        
        elif column_analysis.data_category == DataCategory.CONTACT:
            if 'email' in column_analysis.name.lower():
                schema['check_with'] = 'email_domain'
            elif 'phone' in column_analysis.name.lower():
                schema['check_with'] = 'phone_format'
        
        elif column_analysis.data_category == DataCategory.ADDRESS:
            if 'postal' in column_analysis.name.lower() or 'zip' in column_analysis.name.lower():
                schema['check_with'] = 'postal_code'
        
        if column_analysis.custom_values:
            schema['allowed'] = column_analysis.custom_values
        
        return schema
    
    def build_table_schema(self, table_analysis: TableAnalysis, locale: str = 'en_US') -> Dict[str, Any]:
        schema = {}
        
        for column in table_analysis.columns:
            schema[column.name] = self._build_column_schema(column, locale)
        
        return schema
    
    def build_schema_from_analysis(self, analysis_result: SchemaAnalysisResult, locale: str = 'en_US') -> Dict[str, Dict[str, Any]]:
        schemas = {}
        
        for table_analysis in analysis_result.tables:
            schemas[table_analysis.name] = self.build_table_schema(table_analysis, locale)
        
        return schemas


class PydanticModelFactory:
    
    def __init__(self):
        self.created_models = {}
    
    def _get_pydantic_type(self, column_analysis: ColumnAnalysis):
        type_mapping = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'datetime': datetime,
            'date': date,
            'time': time,
            'uuid': str,
            'decimal': float,
            'text': str,
            'json': dict,
            'list': list
        }
        
        base_type = type_mapping.get(column_analysis.python_type, str)
        
        if column_analysis.nullable:
            return Optional[base_type]
        return base_type
    
    def _create_field(self, column_analysis: ColumnAnalysis):
        field_type = self._get_pydantic_type(column_analysis)
        field_kwargs = {}
        
        if column_analysis.max_length:
            field_kwargs['max_length'] = column_analysis.max_length
        
        if column_analysis.pattern_detected:
            patterns = {
                'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                'phone': r'^\+?[\d\s\-\(\)]+$',
                'url': r'^https?://[^\s/$.?#].[^\s]*$',
                'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            }
            if column_analysis.pattern_detected in patterns:
                field_kwargs['pattern'] = patterns[column_analysis.pattern_detected]
        
        if column_analysis.default_value is not None:
            field_kwargs['default'] = column_analysis.default_value
        
        return Field(**field_kwargs) if field_kwargs else None
    
    def create_table_model(self, table_analysis: TableAnalysis) -> BaseModel:
        if table_analysis.name in self.created_models:
            return self.created_models[table_analysis.name]
        
        fields = {}
        for column in table_analysis.columns:
            field_type = self._get_pydantic_type(column)
            field_def = self._create_field(column)
            
            if field_def:
                fields[column.name] = (field_type, field_def)
            else:
                fields[column.name] = (field_type, ...)
        
        model = create_model(f"{table_analysis.name}Model", **fields)
        self.created_models[table_analysis.name] = model
        return model
    
    def create_models_from_analysis(self, analysis_result: SchemaAnalysisResult) -> Dict[str, BaseModel]:
        models = {}
        
        for table_analysis in analysis_result.tables:
            models[table_analysis.name] = self.create_table_model(table_analysis)
        
        return models


class DataValidator:
    
    def __init__(self, 
                 analysis_result: SchemaAnalysisResult,
                 validation_level: ValidationLevel = ValidationLevel.STANDARD,
                 locale: str = 'en_US',
                 enable_sampling: bool = True,
                 sample_size: int = 10000):
        
        self.analysis_result = analysis_result
        self.validation_level = validation_level
        self.locale = locale
        self.enable_sampling = enable_sampling
        self.sample_size = sample_size
        
        self.business_rules = BusinessRuleValidator()
        self.schema_builder = CerberusSchemaBuilder(self.business_rules)
        self.model_factory = PydanticModelFactory()
        
      
        self.cerberus_schemas = self.schema_builder.build_schema_from_analysis(analysis_result, locale)
        self.pydantic_models = self.model_factory.create_models_from_analysis(analysis_result)
        
        self.validation_issues = []
        self.quality_metrics = {}
        
        self.logger = logging.getLogger(__name__)
    
    def _should_sample_data(self, data_size: int) -> bool:
        if not self.enable_sampling:
            return False
        
        return data_size > self.sample_size and self.validation_level != ValidationLevel.STRICT
    
    def _sample_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(data) <= self.sample_size:
            return data
        
        import random
        random.seed(42)
        return random.sample(data, self.sample_size)
    
    def _validate_with_pydantic(self, table_name: str, records: List[Dict[str, Any]]) -> List[ValidationIssue]:
        issues = []
        model = self.pydantic_models.get(table_name)
        
        if not model:
            issues.append(ValidationIssue(
                table_name=table_name,
                column_name=None,
                row_index=None,
                issue_type="missing_model",
                severity=ValidationResult.FAIL,
                message=f"No Pydantic model found for table {table_name}"
            ))
            return issues
        
        validation_data = self._sample_data(records) if self._should_sample_data(len(records)) else records
        
        for idx, record in enumerate(validation_data):
            try:
                model(**record)
            except ValidationError as e:
                for error in e.errors():
                    field_name = '.'.join(str(loc) for loc in error['loc'])
                    issues.append(ValidationIssue(
                        table_name=table_name,
                        column_name=field_name,
                        row_index=idx,
                        issue_type="pydantic_validation",
                        severity=ValidationResult.FAIL,
                        message=error['msg'],
                        actual_value=record.get(field_name),
                        suggestion="Check data type and format requirements"
                    ))
        
        return issues
    
    def _validate_with_cerberus(self, table_name: str, records: List[Dict[str, Any]]) -> List[ValidationIssue]:
        issues = []
        schema = self.cerberus_schemas.get(table_name)
        
        if not schema:
            issues.append(ValidationIssue(
                table_name=table_name,
                column_name=None,
                row_index=None,
                issue_type="missing_schema",
                severity=ValidationResult.FAIL,
                message=f"No Cerberus schema found for table {table_name}"
            ))
            return issues
        
        class CustomValidator(Validator):
            def _check_with_email_domain(self, field, value):
                if value and not self.business_rules.validate_email_domain(value):
                    self._error(field, f"Invalid email domain: {value}")
            
            def _check_with_phone_format(self, field, value):
                if value and not self.business_rules.validate_phone_format(value, self.locale.split('_')[1] if '_' in self.locale else None):
                    self._error(field, f"Invalid phone format: {value}")
            
            def _check_with_postal_code(self, field, value):
                if value and not self.business_rules.validate_postal_code(value, self.locale.split('_')[1] if '_' in self.locale else None):
                    self._error(field, f"Invalid postal code: {value}")
            
            def _check_with_credit_card(self, field, value):
                if value and not self.business_rules.validate_credit_card(value):
                    self._error(field, f"Invalid credit card: {value}")
        
        CustomValidator.business_rules = self.business_rules
        CustomValidator.locale = self.locale
        
        validator = CustomValidator(schema)
        validation_data = self._sample_data(records) if self._should_sample_data(len(records)) else records
        
        for idx, record in enumerate(validation_data):
            if not validator.validate(record):
                for field, errors in validator.errors.items():
                    for error in errors:
                        issues.append(ValidationIssue(
                            table_name=table_name,
                            column_name=field,
                            row_index=idx,
                            issue_type="cerberus_validation",
                            severity=ValidationResult.FAIL,
                            message=error,
                            actual_value=record.get(field),
                            suggestion="Check business rule compliance"
                        ))
        
        return issues
    
    def _validate_referential_integrity(self, generated_data: Dict[str, List[Dict[str, Any]]]) -> List[ValidationIssue]:
        """Validate foreign key relationships"""
        issues = []
        
        for table_analysis in self.analysis_result.tables:
            table_name = table_analysis.name
            if table_name not in generated_data:
                continue
            
            for column in table_analysis.columns:
                if column.foreign_key:
                    ref_table = column.foreign_key['referenced_table']
                    ref_column = column.foreign_key['referenced_column']
                    
                    if ref_table not in generated_data:
                        issues.append(ValidationIssue(
                            table_name=table_name,
                            column_name=column.name,
                            row_index=None,
                            issue_type="missing_reference_table",
                            severity=ValidationResult.FAIL,
                            message=f"Referenced table {ref_table} not found in generated data"
                        ))
                        continue
                    
                    ref_values = set()
                    for ref_record in generated_data[ref_table]:
                        if ref_column in ref_record and ref_record[ref_column] is not None:
                            ref_values.add(ref_record[ref_column])
                    
                    for idx, record in enumerate(generated_data[table_name]):
                        fk_value = record.get(column.name)
                        if fk_value is not None and fk_value not in ref_values:
                            issues.append(ValidationIssue(
                                table_name=table_name,
                                column_name=column.name,
                                row_index=idx,
                                issue_type="invalid_foreign_key",
                                severity=ValidationResult.FAIL,
                                message=f"Foreign key value {fk_value} not found in {ref_table}.{ref_column}",
                                actual_value=fk_value,
                                suggestion=f"Ensure all foreign key values exist in the referenced table"
                            ))
        
        return issues
    
    def _validate_unique_constraints(self, generated_data: Dict[str, List[Dict[str, Any]]]) -> List[ValidationIssue]:
        issues = []
        
        for table_analysis in self.analysis_result.tables:
            table_name = table_analysis.name
            if table_name not in generated_data:
                continue
            
            for column in table_analysis.columns:
                if column.unique:
                    values = []
                    for record in generated_data[table_name]:
                        value = record.get(column.name)
                        if value is not None:
                            values.append(value)
                    
                    value_counts = Counter(values)
                    for value, count in value_counts.items():
                        if count > 1:
                            issues.append(ValidationIssue(
                                table_name=table_name,
                                column_name=column.name,
                                row_index=None,
                                issue_type="duplicate_unique_value",
                                severity=ValidationResult.FAIL,
                                message=f"Duplicate unique value '{value}' found {count} times",
                                actual_value=value,
                                suggestion="Ensure unique constraint is properly enforced"
                            ))
        
        return issues
    
    def _calculate_quality_metrics(self, generated_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics"""
        metrics = {
            'completeness': {},
            'consistency': {},
            'accuracy': {},
            'validity': {},
            'uniqueness': {},
            'overall_score': 0.0
        }
        
        total_records = sum(len(records) for records in generated_data.values())
        total_fields = 0
        null_fields = 0
        
        for table_analysis in self.analysis_result.tables:
            table_name = table_analysis.name
            if table_name not in generated_data:
                continue
            
            table_records = generated_data[table_name]
            table_metrics = {
                'completeness': 0.0,
                'consistency': 0.0,
                'accuracy': 0.0,
                'validity': 0.0
            }
            
            for column in table_analysis.columns:
                column_values = [record.get(column.name) for record in table_records]
                total_fields += len(column_values)
                null_count = sum(1 for v in column_values if v is None)
                null_fields += null_count
                
                completeness = 1.0 - (null_count / len(column_values)) if column_values else 0.0
                
                consistency = 1.0  
                if column.pattern_detected and column_values:
                    pattern_matches = 0
                    for value in column_values:
                        if value is not None:
                            if column.pattern_detected == 'email' and '@' in str(value):
                                pattern_matches += 1
                            elif column.pattern_detected == 'phone' and any(c.isdigit() for c in str(value)):
                                pattern_matches += 1
                            else:
                                pattern_matches += 1
                    consistency = pattern_matches / len([v for v in column_values if v is not None])
                
                table_metrics['completeness'] += completeness
                table_metrics['consistency'] += consistency
            
            column_count = len(table_analysis.columns)
            if column_count > 0:
                table_metrics['completeness'] /= column_count
                table_metrics['consistency'] /= column_count
                table_metrics['accuracy'] = 1.0 - (len([i for i in self.validation_issues if i.table_name == table_name]) / len(table_records))
                table_metrics['validity'] = table_metrics['accuracy'] 
            
            metrics['completeness'][table_name] = table_metrics['completeness']
            metrics['consistency'][table_name] = table_metrics['consistency']
            metrics['accuracy'][table_name] = table_metrics['accuracy']
            metrics['validity'][table_name] = table_metrics['validity']
        
        if generated_data:
            metrics['overall_completeness'] = 1.0 - (null_fields / total_fields) if total_fields > 0 else 0.0
            metrics['overall_consistency'] = statistics.mean(metrics['consistency'].values()) if metrics['consistency'] else 0.0
            metrics['overall_accuracy'] = statistics.mean(metrics['accuracy'].values()) if metrics['accuracy'] else 0.0
            metrics['overall_validity'] = statistics.mean(metrics['validity'].values()) if metrics['validity'] else 0.0
            
            metrics['overall_score'] = (
                metrics['overall_completeness'] * 0.3 +
                metrics['overall_consistency'] * 0.25 +
                metrics['overall_accuracy'] * 0.25 +
                metrics['overall_validity'] * 0.2
            )
        
        return metrics
    
    def validate_generated_data(self, generated_data: Dict[str, List[Dict[str, Any]]]) -> ValidationReport:
        start_time = datetime.now()
        self.validation_issues = []
        
        self.logger.info(f"Starting validation of {len(generated_data)} tables...")
        
        for table_name, records in generated_data.items():
            self.logger.info(f"Validating table {table_name} with {len(records)} records...")
            
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.STANDARD]:
                pydantic_issues = self._validate_with_pydantic(table_name, records)
                self.validation_issues.extend(pydantic_issues)
            
            cerberus_issues = self._validate_with_cerberus(table_name, records)
            self.validation_issues.extend(cerberus_issues)
        
        if self.validation_level == ValidationLevel.STRICT:
            ref_issues = self._validate_referential_integrity(generated_data)
            self.validation_issues.extend(ref_issues)
            
            unique_issues = self._validate_unique_constraints(generated_data)
            self.validation_issues.extend(unique_issues)
        
        self.quality_metrics = self._calculate_quality_metrics(generated_data)
        
        end_time = datetime.now()
        validation_duration = (end_time - start_time).total_seconds()
        
        total_records = sum(len(records) for records in generated_data.values())
        issues_by_severity = Counter(issue.severity for issue in self.validation_issues)
        issues_by_table = Counter(issue.table_name for issue in self.validation_issues)
        issues_by_type = Counter(issue.issue_type for issue in self.validation_issues)
        
        quality_score = self.quality_metrics.get('overall_score', 0.0)
        
        report = ValidationReport(
            total_records=total_records,
            total_issues=len(self.validation_issues),
            issues_by_severity=dict(issues_by_severity),
            issues_by_table=dict(issues_by_table),
            issues_by_type=dict(issues_by_type),
            detailed_issues=self.validation_issues,
            quality_score=quality_score,
            validation_timestamp=start_time,
            validation_duration=validation_duration,
            quality_metrics=self.quality_metrics
        )
        
        self.logger.info(f"Validation finished in {validation_duration:.2f} seconds. Found {len(self.validation_issues)} issues.")
        
        return report

def export_validation_report(report: ValidationReport, output_dir: str = 'src/output/validation_reports'):
    """
    Exports the validation report to various formats.

    Args:
        report: The ValidationReport object to export.
        output_dir: The directory to save the reports in.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = report.validation_timestamp.strftime('%Y%m%d_%H%M%S')
    base_filename = f"validation_report_{timestamp}"

    # --- Export Summary to TXT ---
    summary_path = output_path / f"{base_filename}_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("========================================\n")
        f.write("      Data Validation Summary Report      \n")
        f.write("========================================\n\n")
        f.write(f"Timestamp: {report.validation_timestamp.isoformat()}\n")
        f.write(f"Duration: {report.validation_duration:.2f} seconds\n")
        f.write(f"Overall Quality Score: {report.quality_score:.2%}\n\n")
        f.write(f"--- Totals ---\n")
        f.write(f"Records Processed: {report.total_records}\n")
        f.write(f"Issues Found: {report.total_issues}\n\n")
        f.write(f"--- Issues by Severity ---\n")
        for severity, count in report.issues_by_severity.items():
            f.write(f"- {severity.name}: {count}\n")
        f.write(f"\n--- Issues by Type ---\n")
        for issue_type, count in report.issues_by_type.items():
            f.write(f"- {issue_type}: {count}\n")

    print(f"Validation summary saved to {summary_path}")

    if report.detailed_issues:
        # --- Export Detailed Issues to JSON ---
        json_path = output_path / f"{base_filename}_details.json"
        # Convert dataclasses to dicts for JSON serialization
        issues_as_dicts = [issue.__dict__ for issue in report.detailed_issues]
        for issue in issues_as_dicts:
            issue['severity'] = issue['severity'].name # Convert enum to string
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(issues_as_dicts, f, indent=2, default=str)
        print(f"Detailed issues JSON report saved to {json_path}")

        # --- Export Detailed Issues to CSV ---
        csv_path = output_path / f"{base_filename}_details.csv"
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if report.detailed_issues:
                writer = csv.DictWriter(f, fieldnames=report.detailed_issues[0].__dict__.keys())
                writer.writeheader()
                writer.writerows(issues_as_dicts)
        print(f"Detailed issues CSV report saved to {csv_path}")