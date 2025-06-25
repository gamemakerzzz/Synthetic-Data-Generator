from re import L
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass
import random
import json
from datetime import datetime, date, time
import uuid
from pathlib import Path


try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False
    print("Warning: Faker not available. Install with : pip install faker")

try:
    from mimesis import Person,Address, Generic, Datetime, Text, Internet, Finance
    from mimesis.locales import Locale
    MIMESIS_AVAILABLE = True
except ImportError:
    MIMESIS_AVAILABLE = False
    print("Warning: Mimesis not available. Install with : pip install mimesis")

from src.core.schema_analyzer import SchemaAnalysisResult, analyze_database_schema, get_generation_config_for_data_generator

@dataclass
class GenerationContext:
    generated_data: Dict[str, List[Dict[str, Any]]]
    primary_key_pools: Dict[str, Dict[str, Set[Any]]]
    current_locale = str
    faker_instance: Optional[Any] = None
    mimesis_providers: Optional[Dict[str, Any]] = None


class DataGenerator:
    def __init__(self, config: Optional[Dict] = None, locale: str = 'en_US'):
        self.config= config or {}
        self.locale  = locale
        self.context = GenerationContext(
            generated_data={},
            primary_key_pools={},
            current_locale=locale
        )

        self._init_faker()
        self._init_mimesis()
        self._init_library_preferences()

    def _init_faker(self):
        if not FAKER_AVAILABLE:
            return
        
        locale_mapping = {
        'TN': 'ar_AA',
        'EN': 'en_US',
        'FR': 'fr_FR',
        'en_US': 'en_US',
        'fr_FR': 'fr_FR'
        }

        faker_locale = locale_mapping.get(self.locale, 'en_US')
        try:
            self.context.faker_instance = Faker(faker_locale)
        except AttributeError:
            self.context.faker_instance = Faker('en_US')

    def _init_mimesis(self):
        if not MIMESIS_AVAILABLE:
            return
        
        locale_mapping = {
        'TN': Locale.AR_TN,
        'EN': Locale.EN,
        'FR': Locale.FR,
        'en_US': Locale.EN,
        'fr_FR': Locale.FR
        }

        mimesis_locale = locale_mapping.get(self.locale,Locale.AR_TN)

        self.context.mimesis_providers = {
            'person' : Person(mimesis_locale),
            'address': Address(mimesis_locale),
            'datetime': Datetime(mimesis_locale),
            'text': Text(mimesis_locale),
            'internet': Internet(),
            'finance': Finance(),
            'generic': Generic(mimesis_locale)
        }
    
    def _init_library_preferences(self):
        self.faker_preferences = {
            'patterns': {
            'company': True,
            'job_title': True,
            'address': True,
            'city': True,
            'state': True,
            'country': True,
            'postal_code': True,
            'phone': True,
            'email': False,
            'url': False,
            'ip_address': False,
            },
            'categories': {
                'BUSINESS': True,
                'ADDRESS': True,
                'CONTACT': False,
                'FINANCIAL': True,
                'PERSONAL': False,
                'TEMPORAL': False,
                'TECHNICAL': False,
                'GENERIC': False,
            },
            'methods': {
                'company': True,
                'occupation': True,
                'address': True,
                'city': True,
                'state': True,
                'country': True,
                'postal_code': True,
                'telephone': True,
                'credit_card_number': True,
                'bank_account': True,
            }
        }
    
    def _should_use_faker(self, column_name: str, column_config: Dict[str, Any])->bool:
        if not FAKER_AVAILABLE:
            return False
        
        pattern = column_config.get('pattern_detected')
        if pattern and self.faker_preferences['patterns'].get(pattern):
            return True
        
        category = column_config.get('data_category', '').upper()
        if self.faker_preferences['categories'].get(category):
            return True
        
        method = column_config.get('method', '')
        if self.faker_prefernces['methods'].get(method):
            return True
        
        return False
    
    def _generate_with_faker(self, column_name: str, column_config: Dict[str, Any])->Any:
        faker = self.context.faker_instance
        method = column_config.get('method', 'word')
        params = column_config.get('params', {})

        method_mapping = {
            'first_name': 'first_name',
            'last_name': 'last_name',
            'full_name': 'name',
            'email': 'email',
            'telephone': 'phone_number',
            'address': 'address',
            'city': 'city',
            'state': 'state',
            'country': 'country',
            'postal_code': 'postcode',
            'company': 'company',
            'occupation': 'job',
            'url': 'url',
            'word': 'word',
            'sentence': 'sentence',
            'text': 'text',
            'integer_number': 'random_int',
            'float_number': 'pyfloat',
            'boolean': 'boolean',
            'datetime': 'date_time',
            'date': 'date',
            'time': 'time',
            'uuid4': 'uuid4',
            'credit_card_number': 'credit_card_number',
        }

        faker_method = method_mapping.get(method,'word')

        try:
            if hasattr(faker, faker_method):
                method_func = getattr(faker, faker_method)

                if faker_method == 'random_int':
                    min_val= params.get('start', 1)
                    max_val = params.get('end', 100000)
                    return method_func(min=min_val, max=max_val)

                elif faker_method == 'pyfloat':
                    min_val = params.get('start', 1)
                    max_val = params.get('end', 1000)
                    precision = params.get('precision', 2)
                    return round(method_func(min_value=min_val, max_value=max_val), precision)

                elif faker_method in ['date_time','date']:
                    start_year = params.get('start', 2020)
                    end_year = params.get('end', 2024)
                    if faker_method == 'date_time':
                        return method_func(start_date=f'{start_year}-01-01', end_date=f'{end_year}-12-31')
                    else:
                        return method_func(start_date=f'{start_year}-01-01', end_date=f'{end_year}-12-31')
                    
                elif faker_method == 'sentence':
                    nb_words =params.get('nb_words', 6)
                    return method_func(nb_words=nb_words)
                
                elif faker_method == 'text':
                    max_nb_chars = params.get('max_nb_chars', 200)
                    return method_func(max_nb_chars=max_nb_chars)
                
                else:
                    return method_func()
                
            else:
                return self._generate_with_mimesis(column_name, column_config)
        except Exception as e:
            print(f"Error generating with Faker for {column_name}: {e}")
            return self._generate_with_mimesis(column_name,column_config)
        
    def _generate_with_mimesis(self, column_name: str, column_config: Dict[str, Any])->Any:
        if not MIMESIS_AVAILABLE:
            return None
        
        provider_type = column_config.get('provider_type','text')
        method = column_config.get('method', 'word')
        params = column_config.get('params', {})

        provider_mapping = {
            'person': 'person',
            'address': 'address',
            'datetime': 'datetime',
            'text': 'text',
            'internet': 'internet',
            'finance': 'finance',
            'numbers': 'numbers',
            'custom': 'generic'
        }

        provider_name = provider_mapping.get(provider_type, 'text')
        provider = self.context.mimesis_providers[provider_name]

        try:
            if hasattr(provider, method):
                method_func = getattr(provider, method)

                if method in ['integer_number', 'float_number']:
                    start = params.get('start', 1)
                    end = params.get('end', 1000)
                    if method == 'float_number':
                        precision = params.get('precision', 2)
                        return round(method_func(start=start,end=end),precision)
                    return method_func(start=start, end=end)
                    
                elif method in ['datetime', 'date']:
                    start = params.get('start', 2020)
                    end = params.get('end', 2024)
                    return method_func(start=start, end=end)
                
                elif method == 'sentence':
                    nb_words = params.get('nb_words', 6)
                    return method_func(nb_words=nb_words)
                
                elif method== 'text':
                    quantity = params.get('quantity', 1)
                    return method_func(quantity=quantity)
                
                elif method=='password':
                    length = params.get('length', 12)
                    return method_func(length=length)
                else:
                    return method_func()
            else:
                return self.context.mimesis_providers['text'].word()

        except Exception as e :
            print(f"Error generating with Mimesis for{column_name}:{e}")
            return "default_value"
    
    def _handle_foreign_key(self, column_name:str, column_config: Dict[str, Any])->Any:
        fk_info = column_config.get('foreign_key')
        if not fk_info:
            return None
        
        referenced_table = fk_info['referenced_table']
        referenced_column = fk_info['referenced_column']

        if referenced_table in self.context.primary_key_pools:
            if referenced_column in self.context.primary_key_pools[referenced_table]:
                available_values = list(self.context.primary_key_pools[referenced_table][referenced_column])
                if available_values:
                    return random.choice(available_values)
                
        print(f"Warning: No available foreign key values for {column_name}->{referenced_table}.{referenced_column}")
        return None
    
    def _handle_custom_values(self, column_config:Dict[str, Any])->Any:
        custom_values = column_config.get('custom_values')
        if custom_values and isinstance(custom_values, list):
            return random.choice(custom_values)
        return None
    
    def _should_generate_null(self, column_config: Dict[str, Any])-> bool:
        if not column_config.get('nullable',True):
            return False
        
        null_probability = column_config.get('null_probability', 0.0)
        return random.random() < null_probability
    
    def _generate_column_value(self, column_name:str, column_config: Dict[str, Any],
                               table_name: str, existing_values:Set[Any]=None)->Any:
        
        if self._should_generate_null(column_config):
            return None
        
        if column_config.get('foreign_key'):
            return self._handle_foreign_key(column_name, column_config)
        
        custom_value = self._handle_custom_values(column_config)
        if custom_value is not None:
            return custom_value
        

        max_attempts = 100
        for attempt in range(max_attempts):
            if self._should_use_faker(column_name,column_config):
                value = self._generate_with_faker(column_name,column_config)
            else:
                 value = self._generate_with_mimesis(column_name,column_config)

            if column_config.get('method') == 'uuid4':
                value = str(uuid.uuid4())
            elif column_config.get('method') == 'boolean':
                value = random.choice([True,False])

            if column_config.get('unique', False) and existing_values is not None:
                if value not in existing_values:
                    existing_values.add(value)
                    return value
            else:
                return value
            
        if column_config.get('unique',False):
            return f"{column_name}_{uuid.uuid4().hex[:8]}"
        
        return value

    def _store_primary_key_values(self, table_name: str,record: Dict[str, Any],
                                  table_config: Dict[str, Any]):
        if table_name not in self.context.primary_key_pools:
            self.context.primary_key_pools[table_name] = {}

        for column_name, column_config in table_config['columns'].items():
            if column_name in record:
                if column_name not in self.context.primary_key_pools[table_name]:
                    self.context.primary_key_pools[table_name][column_name] = set()

                if record[column_name] is not None:
                    self.context.primary_key_pools[table_name][column_name].add(record[column_name])

    def generate_table_data(self, table_name:str, table_config: Dict[str, Any],
                            num_records: int = 100)->List[Dict[str, Any]]:
        records = []
        unique_trackers = {}

        for column_name, column_config in table_config['column'].items():
            if column_config.get('unique',False):
                unique_trackers[column_name] = set()
        
        for i in range(num_records):
            record = {}
            for column_name, column_config in table_config['columns'].items():
                existing_values = unique_trackers.get(column_name)
                value = self._generate_column_value(
                    column_name,column_config,table_name,existing_values
                )
                record[column_name] = value

            records.append(record)
            self._store_primary_key_values(table_name, record, table_config)
        self.context.generated_data[table_name] = records
        return records

    def generate_schema_data(self, analysis_result: SchemaAnalysisResult,
                             records_per_table: Union[int, Dict[str,int]] = 100)->Dict[str, List[Dict[str, Any]]]:
        generation_config = get_generation_config_for_data_generator(analysis_result)

        if isinstance(records_per_table, int):
            table_record_counts = {table: records_per_table for table in generation_config['generation_order']}
        else:
            table_record_counts = records_per_table

        for table_name in generation_config['generation_order']:
            if table_name in generation_config['tables']:
                table_config = generation_config['tables'][table_name]
                num_records = table_record_counts.get(table_name,100)
                print(f"Generating {num_records} records for table : {table_name}")
                self.generate_table_data(table_name, table_config, num_records)
        
        return self.context.generated_data
    
    def export_data(self, output_format: str = 'json', output_dir: str = 'src/output/generated_data'):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if output_format.lower() == 'json':
            for table_name, records in self.context.generated_data.items():
                file_path=output_path/f"{table_name}.json"
                with open(file_path,'w', encoding='utf-8') as f:
                    json.dump(records,f,indent=2,ensure_ascii=False,default=str)
                print(f"Exported {len(records)} records to {file_path}")

        elif output_format.lower() == 'csv':
            import csv
            for table_name, records in self.context.generated_data.items():
                if records:
                    file_path = output_path/f"{table_name}.csv"
                    with open(file_path,'w',newline='',encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=records[0].keys())
                        writer.writeheader()
                        writer.writerows(records)
                    print(f"Exported {len(records)} records to {file_path}")


    def get_generation_summary(self)->Dict[str, Any]:
        summary = {
            'total_tables': len(self.context.generated_data),
            'total_records': sum(len(records)for records in self.context.generated_data),
            'tables': {},
            'generation_timestamp': datetime.now().isoforamt(),
            'locale': self.locale,
            'libraries_used':{
                'faker': FAKER_AVAILABLE,
                'mimesis':MIMESIS_AVAILABLE
            }
        }

        for table_name, records in self.context.generated_data.items():
            summary['tables'][table_name] = {
                'record_count' : len(records),
                'columns' : list(records[0].keys()) if records else []
            }
        return summary
    
    def generate_data_from_schema(config: Optional[Dict] = None,
                                  records_per_table: Union[int, Dict[str,int]]=100,
                                  locale: str = 'en_US',
                                  output_format: str = 'json')->Dict[str, Any]:
        
        print("Analyzing database schema...")
        analysis_result = analyze_database_schema(config)
        generator = DataGenerator(config, locale)

        print(f"Generating data for {len{analysis_result.tables}} tables...")
        generated_data = generator.generate_schema_data(analysis_result, records_per_table)
        generator.export_data(output_format)

        summary = generator.get_generation_summary()
        print(f"\nGeneration complete!")
        print(f"Tables generated: {summary['total_tables']}")
        print(f"Total records: {summary['total_records']}")

        return summary