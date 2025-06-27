import logging
import os
from sqlalchemy import Engine, create_engine, MetaData ,URL
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from tenacity import before_log, retry,stop_after_attempt,wait_fixed,retry_if_exception_type
from typing import  Dict,Optional
output_dir = 'src/output'
os.makedirs(output_dir, exist_ok=True)
logger =logging.getLogger(__name__)

def _mask_password(url_str:str)->str:
    import re
    return re.sub(r'://([^:]+):([^@]+)@', r'://\1:***@', url_str)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry =retry_if_exception_type(OperationalError),
    before_sleep=lambda retry_state: logger.warning(f"Retrying connection (attempt {retry_state.attempt_number}/3)...")
)
def create_database_connection(config: Dict)->Engine:
    url = create_engine_url(config)
    logger.info(f"creating database connection : {_mask_password(str(url))}")
    engine=create_engine(url)
    with engine.connect() as conn:
        logger.info("Database connection successful!!!")
    return engine

def get_database_config():
    db_types ={
        '1': 'mysql',
        '2': 'postgresql',
        '3': 'sqlite',
        '4': 'mssql',
        '5': 'oracle'
    }
    default_ports ={
        'mysql': '3306',
        'postgresql': '5432',
        'mssql': '1433',
        'oracle': '1521'
    }
    drivers = {
        'mysql': '+pymysql',
        'postgresql': '+psycopg2',
        'mssql': '+pyodbc',
        'oracle': '+cx_oracle'
    }
    logger.info("Starting database configuration")
    print("***Configuration de la base de donnees****")
    print("\nChoisissez le type de base de données:")
    print("1. MySQL/MariaDB")
    print("2. PostgreSQL")
    print("3. SQLite")
    print("4. Microsoft SQL Server")
    print("5. Oracle Database")

    choice = input("Votre choix [1] : ").strip() or '1'
    db_type = db_types.get(choice, 'mysql')
    logger.info(f"Selected database type:{db_type}")
    config = {'drivername': db_type,
              'query' : {}}
    if db_type=='sqlite':
        db_file = input("Chemin du fichier SQLite [database.db]: ").strip() or 'database.db'
        config['database'] = db_file
        logger.info(f"SQLite database file: {db_file}")
        return config
    config['username']=input("Username [root]: ").strip() or 'root'
    config['password']= input("Password : ").strip() or ''
    config['host']=input("Hostname [localhost]: ") or 'localhost'
    config['port']=input(f"Port [{default_ports[db_type]}]: ") or default_ports[db_type]

    if db_type == 'oracle':
        print("\nFormat de connexion Oracle:")
        print("1. Service Name (recommandé)")
        print("2. SID (ancienne méthode)")
        oracle_format = input("votre choix [1]: ").strip() or '1'
        if oracle_format == '1':
            service_name= input("service name: ").strip()
            config['query'] ={'service_name': service_name}
            logger.info("Oracle connection using service name")
        else :
            sid =input("SID: ").strip()
            config['database']=sid
            logger.info("Oracle connection using SID")
    elif db_type == 'mssql':
        config['database']=input("Nom de la base de donnes : ").strip()
        print("\nOptions MSSQL:")
        print("1. Windows Authentication")
        print("2. SQL Server Authentication")
        auth_choice = input("Authentication Type [2] : ").strip() or '2'
        if auth_choice == '1':
            config['query'] = {'trusted_connection': 'yes', 'driver': 'ODBC Driver 17 for SQL Server'}
            logger.info("MSSQL using Windows Authentication")
        else:
            config['query'] = {'driver' : 'ODBC Driver 17 for SQL Server'}
            logger.info("MSSQL using SQL Server Authentication")
    else:
        config['database']= input("Nom de la base de donnees : ").strip()
        
    config ['drivername'] += drivers[db_type]
    logger.info(f"Database configuration completed for {config['host']}:{config['port']}")
    return config


def create_engine_url(config):
    db_type=config['drivername'].split('+')[0]
    if db_type=='sqlite':
        url = f"sqlite:///{config['database']}"
        logger.debug(f"SQLite URL created: sqlite:///{config['database']}")  # ADDED: Logging
        return url
    
    if db_type == 'oracle':
        url = URL.create(
            drivername=config['drivername'],
            username=config['username'],
            password=config['password'],
            host=config['host'],
            port=config['port']
        )
        if 'query' in config:
            url = url.update_query_dict(config['query'])
        elif 'database' in config:
            url=url.set(database=config['database'])
        logger.debug("Oracle URL created")
        return url
    url=URL.create(
        drivername=config['drivername'],
        username=config['username'],
        password=config['password'],
        host=config['host'],
        port=config['port'],
        database=config['database']
    )
    if 'query' in config and config['query']:
        url = url.update_query_dict(config['query'])
        logger.debug("URL query parameters applied")  # ADDED: Logging
    
    return url


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(OperationalError),
    before_sleep=lambda retry_state: logger.warning(f"Retrying connection (attempt {retry_state.attempt_number}/3)...")
)
def test_database_connection(config):
    engine= None
    try:
        url =create_engine_url(config)
        logger.info(f"Testing connection: {_mask_password(str(url))}")
        engine=create_engine(url)
        with engine.connect() as _:
            logger.info("Connection test successful")  
            print("\nConnected!!!!")
            metadata=MetaData()
            metadata.reflect(bind=engine)
            if metadata.tables:
                logger.info(f"Found {len(metadata.tables)} tables")  
                print("\nTables disponibles:")
                for table in metadata.sorted_tables:
                    print(f"- {table.name}")
            else:
                logger.warning("No tables found in database")
                print("\nAucune table trouvée dans la base de données.")
            return True
    except OperationalError as e:
        logger.error(f"Connection error: {e}")
        print(f"\n Connection Error: {e}")
        _print_connection_solutions(config['drivername'])
        return False
    
    except ModuleNotFoundError as e :
        logger.error(f"Missing driver module: {e}")
        print(f"\n Inexistent Module : {e}")
        _print_driver_installation(str(e))
        return False
    except SQLAlchemyError as e :
        logger.error(f"SQLAlchemy error: {e}")
        print(f"\n SQLAlchemy Error: {e}")
        return False
    except Exception as e : 
        logger.error(f"Unexpected error: {e}")
        print(f"\n Unexpected Error : {e}")
        return False
    finally:
        if engine:
            engine.dispose()
            logger.debug("Database engine disposed")

def _print_connection_solutions(drivername: str) -> None:
    """Print connection troubleshooting solutions"""
    print("Possible Solutions: ")
    if 'mysql' in drivername:
        print("-verify mySQL")
        print("-Download Driver: pip install pymysql")
        print("-Create DataBase")
    elif 'postgresql' in drivername:
        print("-verify postgresql")
        print("-Download Driver: pip install psycopg2-binary")
    elif 'oracle' in drivername:
        print("-verify Oracle")
        print("-Download Driver: pip install cx_Oracle")
        print("-Verify Service Name/SID")
        print("-Download Oracle Client Instances")
    elif 'mssql' in drivername:  # ADDED: MSSQL solutions
        print("-verify SQL Server is running")
        print("-Download Driver: pip install pyodbc")
        print("-Install ODBC Driver 17 for SQL Server")
        print("-Check Windows/SQL Authentication settings")
    elif 'sqlite' in drivername:
        print(f"verify file path")
        print("Make Sure Directory exist")


def _print_driver_installation(error_str: str) -> None:
    """Print driver installation instructions"""
    print("Download required driver:")
    if 'pymysql' in error_str:
        print("pip install pymysql")
    elif 'psycopg2' in error_str:
        print("pip install psycopg2-binary")
    elif 'cx_Oracle' in error_str:
        print("pip install cx_Oracle")
    elif 'pyodbc' in error_str:
        print("pip install pyodbc")


def install_drivers():
    logger.info("Installing recommended drivers")
    print("Downloading Recommanded drivers...")
    os.system("pip install pymysql psycopg2-binary cx_Oracle pyodbc")
    logger.info("Driver installation completed")

def get_schema_metadata(config: Optional[dict]=None)->Optional[MetaData]:
    if config is None:
        config=get_database_config()
    engine=None
    try:
        url=create_engine_url(config)
        logger.info(f"Getting schema metadata: {_mask_password(str(url))}")
        engine = create_engine(url)

        with engine.connect() as _:
            metadata = MetaData()
            metadata.reflect(bind=engine)
            logger.info(f"Schema metadata extracted: {len(metadata.tables)} tables found")
            return metadata
    except Exception as e:
        logger.error(f"Failed to get schema metadata: {e}")
        return None
    finally:
        if engine:
            engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print("Testing DataBase Connection...")
    print("Note:this application supports MySQL,PostgreSQL,SQLite,SQL server and Oracle\n")
    if input("Do you want to download Recommanded drivers? [O/n] : ").strip().lower() in ['','o','y']:
        install_drivers()
    while True:
        config = get_database_config()
        if test_database_connection(config):
            break
        retry = input("Do you want to retry? [O/n] : ").strip().lower()
        if retry in ['n','no','non']:
            logger.info("Operation cancelled by user")
            print("Operation Cancelled.")
            break