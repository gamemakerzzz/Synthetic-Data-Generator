import json
import csv
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid
from tqdm import tqdm
import time


try:
    from sqlalchemy import create_engine, text, MetaData, Table, inspect
    from sqlalchemy.exc import SQLAlchemyError
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    print("Warning: SQLAlchemy not available. Install with: pip install sqlalchemy")

try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("Warning: openpyxl not available for Excel export. Install with: pip install openpyxl")

try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import psycopg2
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

try:
    import pyodbc
    MSSQL_AVAILABLE = True
except ImportError:
    MSSQL_AVAILABLE = False

try:
    import cx_Oracle
    ORACLE_AVAILABLE = True
except ImportError:
    ORACLE_AVAILABLE = False

from src.utils.database import get_database_config, create_engine_url
from src.core.schema_analyzer import analyze_database_schema
from src.core.data_generator import generate_data_from_schema, DataGenerator

@dataclass
class ExportResult:
    success: bool
    message: str
    exported_files: List[str]= None
    exported_tables: List[str] = None
    total_records: int= 0
    duration: float = 0.0
    errors: List[str] = None

class DataExporter:
    SUPPORTED_FILE_FORMATS = ['json', 'csv', 'excel', 'parquet', 'jsonl']
    SUPPORTED_DATABASES = ['mysql', 'postgresql', 'sqlite', 'mssql', 'oracle']

    def __init__(self, generated_data: Dict[str, List[Dict[str, Any]]],
                 config: Optional[Dict] = None):
        self.generated_data = generated_data
        self.config = config or {}
        self.export_results = {}


    def _export_json(self, output_path:Path, single_file: bool) -> List[str]:
        exported_files = []
        if single_file:
            file_path= output_path / "all_tables.json"
            with open(file_path, 'w', encoding= 'utf-8') as f:
                json.dump(self.generated_data, f, indent=2, ensure_ascii=False,default=str)
            exported_files.append(str(file_path))
        else:
            for table_name, records in self.generated_data.items():
                file_path = output_path / f"{table_name}.json"
                with open(file_path, 'w',encoding='utf_8') as f:
                    json.dump(records, f, indent=2, ensure_ascii=False, default=str)
                exported_files.append(str(file_path))
        return exported_files
    
    def _export_csv(self, output_path: Path)-> List[str]:
        exported_files = []
        for table_name,records in self.generated_data.items():
            if not records:
                continue
            file_path = output_path /f"{table_name}.csv"
            with open(file_path, 'w', newline='', encoding='utf_8') as f :
                writer= csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
            exported_files.append(str(file_path))


        return exported_files
    
    def _export_excel(self, output_path:Path, single_file: bool)->List[str]:
        if not EXCEL_AVAILABLE:
            raise ImportError("openpyxl is required for Excel export. Install with: pip install openpyxl")
        exported_files = []
        if single_file:
            use_multiple_sheets = self._prompt_multiple_sheets()
            file_path = output_path / "all_tables.xlsx"
            if use_multiple_sheets:
                with pd.ExcelWriter(file_path, engine = 'openpyxl') as writer:
                    for table_name, records in self.generated_data.items():
                        if records:
                            df = pd.DataFrame(records)
                            sheet_name = table_name[:31] if len(table_name)> 31 else table_name
                            df.to_excel(writer, sheet_name=sheet_name, index=False)

            else:
                all_data = []
                for table_name,records in self.generated_data.items():
                    for record in records:
                        record_with_table = {'table_name':table_name, **record}
                        all_data.append(record_with_table)

                df = pd.DataFrame(all_data)
                df.to_excel(file_path, index=False)

            exported_files.append(str(file_path))
        else:
            for table_name, records in self.generated_data.items():
                if records:
                    file_path = output_path / f"{table_name}.xlsx"
                    df = pd.DataFrame(records)
                    df.to_excel(file_path, index=False)
                    exported_files.append(str(file_path))

        return exported_files
    
    def _export_parquet(self, output_path: Path)->List[str]:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for Parquet export. Install with: pip install pyarrow")
        
        exported_files = []
        for table_name, records in self.generated_data.items():
            if records:
                file_path = output_path /f"{table_name}.parquet"
                df = pd.DataFrame(records)
                df.to_parquet(file_path, index=False)
                exported_files.append(str(file_path))

        return exported_files
        
    def _export_jsonl(self, output_path: Path)->List[str]:
        exported_files = []
        for table_name ,records in self.generated_data.items():
            if records:
                file_path = output_path / f"{table_name}.jsonl"
                with open(file_path,'w',encoding='utf_8') as f :
                    for record in records:
                        json.dump(record, f , ensure_ascii=False, default=str)
                        f.write('\n')
                exported_files.append(str(file_path))

        return exported_files
    

    def _export_format(self, format_type: str, output_path: Path, single_file: bool)->List[str]:
        exported_files = []

        if format_type == 'json':
            exported_files.extend(self._export_json(output_path,single_file))
        if format_type == 'csv':
            exported_files.extend(self._export_csv(output_path))
        if format_type == 'excel':
            exported_files.extend(self._export_excel(output_path,single_file))
        if format_type == 'parquet':
            exported_files.extend(self._export_parquet(output_path))
        if format_type == 'jsonl':
            exported_files.extend(self._export_jsonl(output_path))

        return exported_files


    def export_to_files(self, output_dir: str = 'src/output/exports',
                        formats: List[str]= None,
                        single_file: bool = False)-> ExportResult:
        start_time = time.time()
        formats = formats or ['json']
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_files = []
        errors = []
        total_records = sum(len(records) for records in self.generated_data.values())

        print(f"Starting file export for {len(self.generated_data)} tables...")
        print(f"Total records to export: {total_records}")

        for format_type in formats:
            if format_type.lower() not in self.SUPPORTED_FILE_FORMATS:
                errors.append(f"Unsupported format: {format_type}")
                continue

            try :
                files = self._export_format(format_type.lower(), output_path,single_file)
                exported_files.extend(files)
                print(f"Successfully exported to {format_type.upper()}")
            except Exception as e:
                error_msg= f"Failed to export to {format_type}: {str(e)}"
                errors.append(error_msg)
                print(f"x{error_msg}")
            
        duration = time.time() - start_time
        return ExportResult(
            success=len(exported_files)>0,
            message=f"Exported {len(exported_files)} files in {duration:.2f}s",
            exported_files=exported_files,
            exported_tables=list(self.generated_data.keys()),
            total_records=total_records,
            duration=duration,
            errors=errors if errors else None
        )
    
    def _prompt_multiple_sheets(self)->bool:
        print("\n" + "="*30)
        print("EXCEL EXPORT OPTIONS")
        print("="*30)
        print("Would you like to create:")
        print("1) Multiple sheets (one per table)")
        print("2) Single sheet (all data combined)")

        while True:
            choice = input("\nEnter your choice (1/2):").strip()
            if choice in ['1', 'multiple']:
                return True
            elif choice in ['2', 'single']:
                return False
            else:
                print("Invalid choice. Please enter '1' for Multiple or '2' for Single.")


    def _determine_insertion_mode(self,engine)->str:
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

        has_existing_data = False
        for table_name in self.generated_data.keys():
            if table_name in existing_tables:
                with engine.connect() as conn:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).fetchone()
                    if result[0]>0:
                        has_existing_data = True
                        break
        if not has_existing_data:
            return 'insert'
        
        print("\n" + "="*50)
        print("EXISTING DATA DETECTED")
        print("="*50)
        print("The database already contains data in one or more target tables.")
        print("\nChoose your insertion mode:")
        print("A) Replace Mode - Delete all existing data and insert new data")
        print("B) Append Mode - Add new data to existing data")

        while True:
            choice = input("\nEnter your choice (A/B): ").upper().strip()
            if choice in ['A','REPLACE']:
                return 'replace'
            elif choice in ['B', 'APPEND']:
                return 'append'
            else:
                print("Invalid choice. Please enter 'A' for Replace or 'B' for Append.")


    def _prompt_duplicate_strategy(self)->str:
        print("\n" + "="*50)
        print("DUPLICATE HANDLING STRATEGY")
        print("="*50)
        print("How should duplicate records be handled?")
        print("1) Keep Old Records - Skip all duplicate new records")
        print("2) Replace with New Records - Remove old duplicates and insert new ones")
        
        while True:
            choice = input("\nEnter your choice (1/2): ").strip()
            if choice in ['1', 'keep']:
                return 'keep_old'
            elif choice in ['2', 'replace']:
                return 'replace_new'
            else:
                print("Invalid choice. Please enter '1' for Keep Old or '2' for Replace.")


    def _bulk_insert_records(self, conn, table_name:str, records: List[Dict])->int :
        if not records:
            return 0
        
        df = pd.DataFrame(records)
        df.to_sql(table_name, conn,if_exists='append', index=False, method='multi')
        return len(records)
    
    def _insert_skip_duplicates(self, conn,table_name: str,records: List[Dict])->int:
        inspector = inspect(conn.engine)
        pk_columns = inspector.get_pk_contraint(table_name)['constrained_columns']
        if not pk_columns:
            return self._bulk_insert_records(conn,table_name,records)
        inserted_count = 0
        for record in records:
            pk_conditions = []
            for pk_col in pk_columns:
                if pk_col in record:
                    pk_conditions.append(f"{pk_col} = :{pk_col}")

            if pk_conditions:
                check_query = f"SELECT COUNT(*) FROM {table_name} WHERE {'AND'.join(pk_conditions)}"
                result = conn.execute(text(check_query),record).fetchone()

                if result[0] == 0:
                    df = pd.DataFrame([record])
                    df.to_sql(table_name, conn, if_exists='append', index=False)
                    inserted_count += 1
        return inserted_count
    
    def _insert_replace_duplicates(self,conn ,table_name:str,records: List[Dict])->int:
        inspector = inspect(conn.engine)
        pk_columns = inspector.get_pk_constraint(table_name)['constrained_columns']

        if not pk_columns:
            return self._bulk_insert_records(conn, table_name,records)
        
        inserted_count = 0
        for record in records:
            pk_conditions = []
            for pk_col in pk_columns:
                if pk_col in record:
                    pk_conditions.append(f"{pk_col} = :{pk_col}")
            if pk_conditions:
                delete_query = f"DELETE FROM {table_name} WHER {'AND'.join(pk_conditions)}"
                conn.execute(text(delete_query),record)

            df= pd.DataFrame([record])
            df.to_sql(table_name, conn, if_exists='append', index=False)
            inserted_count +=1
        return inserted_count

    def _process_table_insertion(self, conn ,table_name:str, records: List[Dict],
                                  insertion_mode: str , duplicate_strategy: Optional[str],
                                  error_handling: str)->int:
        if insertion_mode == 'replace':
            conn.execute(text(f"DELETE FROM {table_name}"))
            print(f"Cleared existing data from {table_name}")
        
        if insertion_mode == 'insert' or insertion_mode == 'replace':
            return self._bulk_insert_records(conn, table_name, records)
        
        elif insertion_mode == 'append':
            if duplicate_strategy == 'keep_old':
                return self._insert_skip_duplicates(conn, table_name, records)
            elif duplicate_strategy == 'replace_new':
                return self._insert_replace_duplicates
        return 0
    

    def _insert_data_to_database(self, engine,insertion_mode:str,
                                 duplicate_strategy: Optional[str],
                                 error_handling:str)-> ExportResult:
        start_time = time.time()
        errors = []
        total_records = sum(len(records) for records in self.generated_data.values())
        inserted_records = 0
        print(f"\nStarting database insertion...")
        print(f"Mode: {insertion_mode.upper()}")
        if duplicate_strategy:
            print(f"Duplicate Strategy: {duplicate_strategy.upper()}")
        print(f"Total records to insert: {total_records}")

        try :
            with engine.begin() as conn:
                schmea_analysis = analyze_database_schema(self.config)
                table_order = schmea_analysis.generation_order

                tables_to_process = [t for t in table_order if t in self.generated_data]

                for table_name in self.generated_data:
                    if table_name not in tables_to_process:
                        tables_to_process.append(table_name)

                progress_bar = tqdm(total = total_records, desc = "Inserting records",unit = "records")
                for table_name in tables_to_process:
                    records = self.generated_data[table_name]
                    if not records:
                        continue
                    try :
                        table_inserted = self._process_table_insertion(
                            conn, table_name, records, insertion_mode,
                            duplicate_strategy, error_handling
                        )
                        inserted_records += table_inserted
                        progress_bar.update(table_inserted)

                    except Exception as e:
                        error_msg = f"Error inserting table {table_name}: {str(e)}"
                        errors.append(error_msg)

                        if error_handling == 'stop':
                            progress_bar.close()
                            raise Exception(error_msg)
                        elif error_handling == 'continue':
                            print(f"{error_msg} (continuing...)")
                            continue

                    progress_bar.close()

        except Exception as e:
            return ExportResult(
                success=False,
                message=f"Database insertion failed: {str(e)}",
                errors = errors + [str(e)]
            )
        duration = time.time() - start_time
        return ExportResult(
            success=True,
            message=f"succesfully inserted {inserted_records} records in {duration:.2f}s",
            exported_tables=list(self.generated_data.keys()),
            total_records= inserted_records,
            duration=duration,
            errors = errors if errors else None
        )

                    


    def export_to_database(self, db_config:Optional[Dict]= None,
                           error_handling:str = 'stop')->ExportResult:
        if not SQLALCHEMY_AVAILABLE:
            return ExportResult(
                success=False,
                message="SQLAlchemy is required for database export",
                errors = ["SQLAlchemy not available"]
            )
        start_time = time.time()
        db_config = db_config or get_database_config()

        if not db_config:
            return ExportResult(
                success=False,
                message="No database configuration provided",
                errors = ["Database configuration missing"]
            )
        try:
            engine_url = create_engine_url(db_config)
            engine = create_engine(engine_url)

            insertion_mode = self._determine_insertion_mode(engine)
            duplicate_strategy = None
            self.config = db_config
            if insertion_mode == 'append':
                duplicate_strategy = self._prompt_duplicate_strategy()
            return self._insert_data_to_database(engine, insertion_mode, duplicate_strategy, error_handling)
        except Exception as e :
            return ExportResult(
                success=False,
                message=f"Database export failed: {str(e)}",
                errors = [str(e)]
            )
        
    def export_summary(self) -> Dict[str, Any]:
        return {
            'total_tables': len(self.generated_data),
            'total_records': sum(len(records) for records in self.generated_data.values()),
            'tables': {
                table_name: {
                    'record_count': len(records),
                    'columns': list(records[0].keys()) if records else []
                }
                for table_name, records in self.generated_data.items()
            },
            'export_timestamp': datetime.now().isoformat(),
            'supported_formats': self.SUPPORTED_FILE_FORMATS,
            'supported_databases': self.SUPPORTED_DATABASES,
            'export_results': [
                {
                    'success': result.success,
                    'message': result.message,
                    'duration': result.duration,
                    'total_records': result.total_records
                }
                for result in self.export_results
            ]
        }

def export_generated_data(generated_data: Dict[str, List[Dict[str, Any]]],
                         export_type: str = 'file',
                         formats: List[str] = None,
                         output_dir: str = 'src/output/exports',
                         db_config: Optional[Dict] = None,
                         **kwargs) -> ExportResult:
    exporter = DataExporter(generated_data)
    
    if export_type.lower() == 'file':
        formats = formats or ['json']
        single_file = kwargs.get('single_file', False)
        return exporter.export_to_files(output_dir, formats, single_file)
    
    elif export_type.lower() == 'database':
        error_handling = kwargs.get('error_handling', 'stop')
        return exporter.export_to_database(db_config, error_handling)
    
    else:
        return ExportResult(
            success=False,
            message=f"Unsupported export type: {export_type}",
            errors=[f"Export type must be 'file' or 'database', got '{export_type}'"]
        )

if __name__ == "__main__":
    try:
        print("\n" + "=" * 50)
        print("DATA GENERATION OPTIONS")
        print("=" * 50)
        records_input = input("Enter number of records per table (default 50): ").strip()
        records_per_table = int(records_input) if records_input.isdigit() else 50
        locale = input("Enter locale (e.g., en_US, fr_FR, ar_TN, default en_US): ").strip() or 'en_US'

        print("Analyzing database schema...")
        config = get_database_config()
        analysis_result = analyze_database_schema(config)
        generator = DataGenerator(locale=locale)
        print(f"Generating data for {len(analysis_result.tables)} tables...")
        generated_data = generator.generate_schema_data(analysis_result, records_per_table)
        summary = generator.get_generation_summary()
        print(f"\nGeneration complete!")
        print(f"Tables generated: {summary['total_tables']}")
        print(f"Total records: {summary['total_records']}")

     
        print("\n" + "=" * 50)
        print("DATA EXPORT OPTIONS")
        print("=" * 50)
        print("Choose an export option:")
        print("1) Export to files (JSON, CSV, Excel, etc.)")
        print("2) Insert into database")
        choice = input("\nEnter your choice (1/2): ").strip()

        if choice == '1':
            print("\nAvailable file formats: json, csv, excel, parquet, jsonl")
            formats_input = input("Enter desired formats (comma-separated, e.g., json,csv): ").strip()
            formats = [f.strip() for f in formats_input.split(',')] if formats_input else ['json']
            single_file = input("Export as a single file? (y/n): ").strip().lower() == 'y'
            result = export_generated_data(
                generated_data,
                export_type='file',
                formats=formats,
                single_file=single_file
            )
            print(f"Export result: {result.message}")
            if result.exported_files:
                print(f"Files created: {result.exported_files}")
        elif choice == '2':
            result = export_generated_data(
                generated_data,
                export_type='database',
                error_handling='stop',
                db_config=config
            )
            print(f"Export result: {result.message}")
            if result.exported_tables:
                print(f"Tables inserted: {result.exported_tables}")
        else:
            print("Invalid choice. Please run again and select '1' or '2'.")
    except Exception as e:
        print(f"Error during data generation or export: {e}")