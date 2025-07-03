from sqlalchemy import MetaData, Table, Column, Column
from sqlalchemy.types import (
    String, Integer, Float, Boolean, DateTime, Date, Time,
    Text, DECIMAL, NUMERIC, BigInteger, SmallInteger,
    CHAR, VARCHAR, JSON, ARRAY
)
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import re
from datetime import datetime, date
import json

from src.utils.database import get_schema_metadata

class DataCategory(Enum):
    PERSONAL = "personal"
    CONTACT = "contact"
    ADDRESS = "address"
    BUSINESS = "business"
    FINANCIAL = "financial"
    TEMPORAL = "temporal"
    TECHNICAL = "technical"
    GENERIC = "generic"

class ProviderType(Enum):
    PERSON = "person"
    ADDRESS = "address"
    DATETIME = "datetime"
    TEXT = "text"
    INTERNET = "internet"
    FINANCE = "finance"
    NUMBERS = "numbers"
    CUSTOM = "custom"


@dataclass
class ColumnAnalysis:
    name : str
    sql_type: str
    python_type : str
    nullable : bool
    unique : bool
    primary_key : bool
    foreign_key : Optional[Dict[str,str]]
    max_length : Optional[int]
    default_value: Any

    data_category: DataCategory
    provider_type: ProviderType
    mimesis_method: str
    method_params: Dict[str,Any]
    custom_values: Optional[List[Any]]
    null_probability: float
    pattern_detected: Optional[str]
    constraints: List[str]

    def to_dict(self)-> Dict[str,Any]:
        result = asdict(self)

        result['data_category'] = self.data_category.value
        result['provider_type'] = self.provider_type.value
        return result


@dataclass   
class TableAnalysis:
    name: str
    columns: List[ColumnAnalysis]
    primary_keys: List[str]
    foreign_keys: Dict[str,Dict[str,str]]
    relationships: List[Dict[str,Any]]
    estimated_complexity: str
    generation_order_priority: int
    
    def to_dict(self) -> Dict[str,Any]:
        return{
            'name': self.name,
                'columns': [col.to_dict() for col in self.columns],
                'primary_keys': self.primary_keys,
                'foreign_keys': self.foreign_keys,
                'relationships': self.relationships,
                'estimated_complexity': self.estimated_complexity,
                'generation_order_priority': self.generation_order_priority
        }

@dataclass   
class SchemaAnalysisResult:
    tables: List[TableAnalysis]
    relationships_graph: Dict[str, List[str]]
    generation_order: List[str]
    suggested_locales: List[str]
    total_complexity_score: int
    analysis_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tables': [table.to_dict() for table in self.tables],
            'relationships_graph': self.relationships_graph,
            'generation_order': self.generation_order,
            'suggested_locales': self.suggested_locales,
            'total_complexity_score': self.total_complexity_score,
            'analysis_metadata': self.analysis_metadata
        }

class SchemaAnalyser:
    def __init__(self,default_locale:str='TN'):
        self.default_locale = default_locale
        self.supported_locales = ['TN', 'EN', 'FR']
        self.column_patterns = {
            'email': {
                'patterns': [r'.*email.*', r'.*mail.*', r'.*e_mail.*'],
                'category': DataCategory.CONTACT,
                'provider': ProviderType.PERSON,
                'method': 'email',
                'params': {}
            },
            'phone': {
                'patterns': [r'.*phone.*', r'.*tel.*', r'.*mobile.*', r'.*cell.*'],
                'category': DataCategory.CONTACT,
                'provider': ProviderType.PERSON,
                'method': 'telephone',
                'params': {}
            },
            'first_name': {
                'patterns': [r'.*first.*name.*', r'.*fname.*', r'.*prenom.*'],
                'category': DataCategory.PERSONAL,
                'provider': ProviderType.PERSON,
                'method': 'first_name',
                'params': {}
            },
            'last_name': {
                'patterns': [r'.*last.*name.*', r'.*lname.*', r'.*nom.*', r'.*surname.*'],
                'category': DataCategory.PERSONAL,
                'provider': ProviderType.PERSON,
                'method': 'last_name',
                'params': {}
            },
            'full_name': {
                'patterns': [r'^name$', r'.*full.*name.*', r'.*complete.*name.*'],
                'category': DataCategory.PERSONAL,
                'provider': ProviderType.PERSON,
                'method': 'full_name',
                'params': {}
            },
            'username': {
                'patterns': [r'.*username.*', r'.*user.*name.*', r'.*login.*'],
                'category': DataCategory.TECHNICAL,
                'provider': ProviderType.PERSON,
                'method': 'username',
                'params': {}
            },
            'password': {
                'patterns': [r'.*password.*', r'.*pwd.*', r'.*pass.*'],
                'category': DataCategory.TECHNICAL,
                'provider': ProviderType.PERSON,
                'method': 'password',
                'params': {'length': 12}
            },
            
            'address': {
                'patterns': [r'.*address.*', r'.*addr.*', r'.*adresse.*'],
                'category': DataCategory.ADDRESS,
                'provider': ProviderType.ADDRESS,
                'method': 'address',
                'params': {}
            },
            'city': {
                'patterns': [r'.*city.*', r'.*ville.*', r'.*town.*'],
                'category': DataCategory.ADDRESS,
                'provider': ProviderType.ADDRESS,
                'method': 'city',
                'params': {}
            },
            'state': {
                'patterns': [r'.*state.*', r'.*province.*', r'.*gouvernorat.*'],
                'category': DataCategory.ADDRESS,
                'provider': ProviderType.ADDRESS,
                'method': 'state',
                'params': {}
            },
            'country': {
                'patterns': [r'.*country.*', r'.*pays.*', r'.*nation.*'],
                'category': DataCategory.ADDRESS,
                'provider': ProviderType.ADDRESS,
                'method': 'country',
                'params': {}
            },
            'postal_code': {
                'patterns': [r'.*postal.*', r'.*zip.*', r'.*code.*postal.*'],
                'category': DataCategory.ADDRESS,
                'provider': ProviderType.ADDRESS,
                'method': 'postal_code',
                'params': {}
            },
            
            'company': {
                'patterns': [r'.*company.*', r'.*entreprise.*', r'.*business.*', r'.*corp.*'],
                'category': DataCategory.BUSINESS,
                'provider': ProviderType.TEXT,
                'method': 'word',  
                'params': {}
            },
            'job_title': {
                'patterns': [r'.*job.*', r'.*title.*', r'.*position.*', r'.*poste.*'],
                'category': DataCategory.BUSINESS,
                'provider': ProviderType.PERSON,
                'method': 'occupation',
                'params': {}
            },
            
            'price': {
                'patterns': [r'.*price.*', r'.*cost.*', r'.*amount.*', r'.*prix.*'],
                'category': DataCategory.FINANCIAL,
                'provider': ProviderType.NUMBERS,
                'method': 'float_number',
                'params': {'start': 1, 'end': 1000, 'precision': 2}
            },
            'salary': {
                'patterns': [r'.*salary.*', r'.*wage.*', r'.*salaire.*'],
                'category': DataCategory.FINANCIAL,
                'provider': ProviderType.NUMBERS,
                'method': 'integer_number',
                'params': {'start': 30000, 'end': 150000}
            },
            
            'url': {
                'patterns': [r'.*url.*', r'.*link.*', r'.*website.*'],
                'category': DataCategory.TECHNICAL,
                'provider': ProviderType.INTERNET,
                'method': 'url',
                'params': {}
            },
            'ip_address': {
                'patterns': [r'.*ip.*', r'.*ip.*address.*'],
                'category': DataCategory.TECHNICAL,
                'provider': ProviderType.INTERNET,
                'method': 'ip_v4',
                'params': {}
            },
            'uuid': {
                'patterns': [r'.*uuid.*', r'.*guid.*'],#, r'.*id.*'
                'category': DataCategory.TECHNICAL,
                'provider': ProviderType.CUSTOM,
                'method': 'uuid4',
                'params': {}
            },
            
            'created_at': {
                'patterns': [r'.*created.*', r'.*date.*creation.*'],
                'category': DataCategory.TEMPORAL,
                'provider': ProviderType.DATETIME,
                'method': 'datetime',
                'params': {'start': 2020, 'end': 2024}
            },
            'updated_at': {
                'patterns': [r'.*updated.*', r'.*modified.*'],
                'category': DataCategory.TEMPORAL,
                'provider': ProviderType.DATETIME,
                'method': 'datetime',
                'params': {'start': 2023, 'end': 2024}
            },
            'birth_date': {
                'patterns': [r'.*birth.*', r'.*born.*', r'.*naissance.*', r'.*dob.*'],
                'category': DataCategory.TEMPORAL,
                'provider': ProviderType.DATETIME,
                'method': 'date',
                'params': {'start': 1950, 'end': 2005}
            }
        }
    def _get_python_type(self,sql_type)->str:
        type_mapping = {
            String: 'str', VARCHAR: 'str', CHAR: 'str', Text: 'str',
            Integer: 'int', BigInteger: 'int', SmallInteger: 'int',
            Float: 'float', DECIMAL: 'float', NUMERIC: 'float',
            Boolean: 'bool',
            DateTime: 'datetime', Date: 'date', Time: 'time',
            JSON: 'dict', ARRAY: 'list'
        }
        for sql_type_class,python_type in type_mapping.items():
            if isinstance(sql_type,sql_type_class):
                return python_type
        return 'str'

    def _detect_column_pattern(self,column_name: str)->Optional[Dict[str,Any]]:
        for pattern_name,pattern_info in self.column_patterns.items():
            for pattern in pattern_info['patterns']:
                if re.match(pattern,column_name,re.IGNORECASE):
                    result = pattern_info.copy()
                    result['pattern_name']= pattern_name
                    return result
        return None


    def _map_type_to_provider(self,column_type,max_length:Optional[int])-> Tuple[DataCategory,ProviderType,str,Dict[str,Any]]:
        if isinstance(column_type,(String,VARCHAR,CHAR,Text)):
            if max_length and max_length <= 10:
                return DataCategory.GENERIC,ProviderType.TEXT,'word',{}
            elif max_length and max_length <=50:
                return DataCategory.GENERIC,ProviderType.TEXT,'sentence',{'nb_words':3}
            else:
                return DataCategory.GENERIC,ProviderType.TEXT,'text',{'quantity':1}
        elif isinstance(column_type, (Integer, BigInteger, SmallInteger)):
            return DataCategory.GENERIC, ProviderType.NUMBERS, 'integer_number', {'start': 1, 'end': 100000}
        
        elif isinstance(column_type, (Float, DECIMAL, NUMERIC)):
            return DataCategory.GENERIC, ProviderType.NUMBERS, 'float_number', {'start': 1, 'end': 1000, 'precision': 2}
        
        elif isinstance(column_type, Boolean):
            return DataCategory.GENERIC, ProviderType.CUSTOM, 'boolean', {}
        
        elif isinstance(column_type, DateTime):
            return DataCategory.TEMPORAL, ProviderType.DATETIME, 'datetime', {'start': 2020, 'end': 2024}
        
        elif isinstance(column_type, Date):
            return DataCategory.TEMPORAL, ProviderType.DATETIME, 'date', {'start': 2020, 'end': 2024}
        
        elif isinstance(column_type, Time):
            return DataCategory.TEMPORAL, ProviderType.DATETIME, 'time', {}
        
        elif isinstance(column_type, JSON):
            return DataCategory.TECHNICAL, ProviderType.CUSTOM, 'json_object', {}
        
        else:
            # Default fallback
            return DataCategory.GENERIC, ProviderType.TEXT, 'word', {}
    
    def _extract_column_constraints(self,column:Column)->List[str]:
        constraints = []
        if not column.nullable:
            constraints.append('NOT NULL')
        if column.unique:
            constraints.append('UNIQUE')
        if column.primary_key:
            constraints.append('PRIMARY KEY')
        if column.foreign_keys:
            constraints.append('FOREIGN KEY')
        if column.default:
            constraints.append(f'DEFAULT {column.default.arg}')   # type: ignore
        if hasattr(column.type,'length')  and column.type.length: # type: ignore
            constraints.append(f'MAX_LENGTH({column.type.length})') # type: ignore
        return constraints
        
    def _analyze_column(self,column:Column)->ColumnAnalysis:
        sql_type = str(column.type)
        python_type = self._get_python_type(column.type)
        nullable = column.nullable
        unique = column.unique or column.primary_key
        primary_key = column.primary_key
        max_length = getattr(column.type,'length',None)
        default_value = column.default.arg if column.default else None # type: ignore
        foreign_key = None
        if column.foreign_keys:
            fk = list(column.foreign_keys)[0]
            foreign_key = {
                'referenced_table' : fk.column.table.name,
                'referenced_column' : fk.column.name
            }
        pattern_info = self._detect_column_pattern(column.name.lower())
        if pattern_info:
            data_category = pattern_info['category']
            provider_type = pattern_info['provider']
            mimesis_method = pattern_info['method']
            method_params = pattern_info['params'].copy()
            pattern_detected = pattern_info.get('pattern_name')
        else:
            data_category, provider_type, mimesis_method, method_params = self._map_type_to_provider(column.type, max_length)
            pattern_detected=None
        null_probability = 0.1 if nullable and not primary_key else 0.0
        constraints = self._extract_column_constraints(column)
        return ColumnAnalysis(
            name=column.name,
            sql_type=sql_type,
            python_type=python_type,
            nullable=column.nullable if column.nullable is not None else True,
            unique=unique,
            primary_key=primary_key,
            foreign_key=foreign_key,
            max_length=max_length,
            default_value=default_value,
            data_category=data_category,
            provider_type=provider_type,
            mimesis_method=mimesis_method,
            method_params=method_params,
            custom_values=None,
            null_probability=null_probability,
            pattern_detected=pattern_detected,
            constraints=constraints
        )


    def _analyze_relationship(self,table:Table)->List[Dict[str,Any]]:
        relationships=[]
        for column in table.columns:
            if column.foreign_keys:
                for fk in column.foreign_keys:
                    relationships.append({
                        'type' : 'foreign_key',
                        'local_column' : column.name,
                        'referenced_table': fk.column.table.name,
                        'referenced_column': fk.column.name,
                        'relationship_type': 'many_to_one'
                    })
        return relationships

    def _calculate_table_complexity(self,table_analysis:TableAnalysis)->int:
        complexity=0
        complexity += len(table_analysis.columns)
        complexity += len(table_analysis.foreign_keys)*2
        for column in table_analysis.columns:
            if column.data_category in [DataCategory.FINANCIAL,DataCategory.TECHNICAL]:
                complexity +=1
            if column.unique and not column.primary_key:
                complexity +=1
            if column.constraints:
                complexity += len(column.constraints)
        return complexity
    

    def _estimate_table_complexity(self,table:Table,column_analyses:List[ColumnAnalysis])->str:
        score=self._calculate_table_complexity(
            TableAnalysis(
                name=table.name,
                columns=column_analyses,
                primary_keys=[],
                foreign_keys={},
                relationships=[],
                estimated_complexity="",
                generation_order_priority=0
            )
        )

        if score<5:
            return "low"
        elif score<15:
            return "medium"
        else:
            return "high"

    def _analyze_table(self,table:Table)->TableAnalysis:
        column_analyses =[]
        primary_keys =[]
        foreign_keys = {}
        for column in table.columns:
            column_analysis = self._analyze_column(column)
            column_analyses.append(column_analysis)
            if column.primary_key:
                primary_keys.append(column.name)
            if column.foreign_keys:
                for fk in column.foreign_keys:
                    foreign_keys[column.name]={
                        'referenced_table': fk.column.table.name,
                        'referenced_column':fk.column.name
                    }
        relationships = self._analyze_relationship(table)
        complexity = self._estimate_table_complexity(table,column_analyses)
        return TableAnalysis(
            name=table.name,
            columns=column_analyses,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            relationships=relationships,
            estimated_complexity=complexity,
            generation_order_priority=0 
        )
    

    def _extract_table_relationships(self,table:Table)->List[str]:
        relationships = []
        for column in table.columns:
            if column.foreign_keys:
                for fk in column.foreign_keys:
                    relationships.append(fk.column.table.name)
        return list(set(relationships))
    
    
    def _calculate_generation_order(self,table_analyses: List[TableAnalysis])-> List[str]:
        dependencies = {}
        all_tables = set()
        for table_analysis in table_analyses:
            table_name=table_analysis.name
            all_tables.add(table_name)
            dependencies[table_name]= set()
            for fk_column,fk_info in table_analysis.foreign_keys.items():
                referenced_table = fk_info['referenced_table']
                if referenced_table != table_name:
                    dependencies[table_name].add(referenced_table)
            
        ordered_tables=[]
        visited=set()
        temp_visited = set()

        def visit(table_name):
            if table_name in temp_visited:
                return
            if table_name in visited:
                return
            temp_visited.add(table_name)
            for dependency in dependencies.get(table_name,[]):
                if dependency in all_tables:
                    visit(dependency)
                
            temp_visited.remove(table_name)
            visited.add(table_name)

            if table_name not in ordered_tables:
                ordered_tables.append(table_name)
            
        for table_name in all_tables:
            if table_name not in visited:
                visit(table_name)
        
        return ordered_tables


    def _suggest_locales(self,table_analyses : List[TableAnalysis])->List[str]:
        locale_scores = {locale:0.0 for locale in self.supported_locales}
        locale_scores[self.default_locale] += 10

        for table_analysis in table_analyses:
            for column in table_analysis.columns:
                if column.data_category in [DataCategory.PERSONAL,DataCategory.ADDRESS]:
                    locale_scores[self.default_locale] += 1
                    if 'EN' in self.supported_locales:
                        locale_scores['EN'] += 0.5
            
        sorted_locales = sorted(locale_scores.items(),key=lambda x:x[1],reverse=True)
        return[locale for locale,score in sorted_locales if score >0]


    def _get_patterns_summary(self,table_analyses:List[TableAnalysis])->Dict[str,int]:
        pattern_counts={}
        for table_analysis in table_analyses:
            for column in table_analysis.columns:
                if column.pattern_detected:
                    pattern_counts[column.pattern_detected]= pattern_counts.get(column.pattern_detected,0)+1
        
        return pattern_counts

    def analyze_schema(self,metadata:MetaData)->SchemaAnalysisResult:
        if not metadata or not metadata.tables:
            raise ValueError("No  metadata or tabls found")
        table_analyses = []
        relationships_graph = {}
        for table_name, table in metadata.tables.items():
            table_analysis = self._analyze_table(table)
            table_analyses.append(table_analysis)
            relationships_graph[table_name] = self._extract_table_relationships(table)


        generation_order = self._calculate_generation_order(table_analyses)
        for i ,table_name in enumerate(generation_order):
            for table_analysis in table_analyses:
                if table_analysis.name == table_name:
                    table_analysis.generation_order_priority= i
                    break

        total_complexity = sum(self._calculate_table_complexity(ta) for ta in table_analyses)
        suggested_locales = self._suggest_locales(table_analyses)

        analysis_metadata = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_tables': len(table_analyses),
            'total_columns': sum(len(ta.columns) for ta in table_analyses),
            'default_locale': self.default_locale,
            'patterns_detected': self._get_patterns_summary(table_analyses)
        }

        return SchemaAnalysisResult(
            tables=table_analyses,
            relationships_graph=relationships_graph,
            generation_order=generation_order,
            suggested_locales=suggested_locales,
            total_complexity_score=total_complexity,
            analysis_metadata=analysis_metadata
        )
    

def analyze_database_schema(config:Optional[Dict]=None) -> SchemaAnalysisResult:
    metadata = get_schema_metadata(config)
    
    if not metadata:
        raise ValueError("Could not retrieve schema metadata")
    
    locale = 'TN'
    if config and 'locale' in config:
        locale = config['locale']

    analyzer = SchemaAnalyser(default_locale=locale)
    
    return analyzer.analyze_schema(metadata)


def export_analysis_config(analysis_result: SchemaAnalysisResult,output_file:str='schema_analysis.json')->None:
    with open(output_file,'w',encoding='utf-8') as f:
        json.dump(analysis_result.to_dict(),f,indent=2,ensure_ascii=False,default=str)



def get_generation_config_for_data_generator(analysis_result:SchemaAnalysisResult)->Dict[str,Any]:
    config = {
        'generation_order' : analysis_result.generation_order,
        'suggested_locales' : analysis_result.suggested_locales,
        'tables': {}
    }
    for table_analysis in analysis_result.tables:
        table_config={
            'priority': table_analysis.generation_order_priority,
            'complexity': table_analysis.estimated_complexity,
            'relationships': table_analysis.relationships,
            'columns': {}
        }
        for column in table_analysis.columns:
            column_config = {
                'provider_type': column.provider_type.value,
                'method': column.mimesis_method,
                'params': column.method_params,
                'nullable': column.nullable,
                'null_probability': column.null_probability,
                'unique': column.unique,
                'constraints': column.constraints,
                'data_category': column.data_category.value
            }

            if column.foreign_key:
                column_config['foreign_key'] = column.foreign_key

            if column.custom_values:
                column_config['custom_values']=column.custom_values

            table_config['columns'][column.name] = column_config
        config['tables'][table_analysis.name] = table_config
    return config

if __name__ == "__main__":
    try:
        analysis_result = analyze_database_schema()
        export_analysis_config(analysis_result, 'src/output/schema_analysis.json')
        generator_config = get_generation_config_for_data_generator(analysis_result)
        
        print(f"Analysis complete!")
        print(f"Tables analyzed: {len(analysis_result.tables)}")
        print(f"Generation order: {analysis_result.generation_order}")
        print(f"Suggested locales: {analysis_result.suggested_locales}")
        print(f"Total complexity score: {analysis_result.total_complexity_score}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")