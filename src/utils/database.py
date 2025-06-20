from sqlalchemy import create_engine, MetaData ,URL
from sqlalchemy.exc import OperationalError, SQLAlchemyError
import os

from tenacity import retry
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
    print("***Configuration de la base de donnees****")
    print("\nChoisissez le type de base de données:")
    print("1. MySQL/MariaDB")
    print("2. PostgreSQL")
    print("3. SQLite")
    print("4. Microsoft SQL Server")
    print("5. Oracle Database")

    choice = input("Votre choix [1] : ").strip() or '1'
    db_type = db_types.get(choice, 'mysql')
    config = {'drivername': db_type,
              'query' : {}}
    if db_type=='sqlite':
        db_file = input("Chemin du fichier SQLite [database.db]: ").strip() or 'database.db'
        config['database'] = db_file
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
        else :
            sid =input("SID: ").strip()
            config['database']=sid
    else:
        config['database']= input("Nom de la base de donnees : ").strip()
        
    config ['drivername'] += drivers[db_type]
    return config


def create_engine_url(config):
    db_type=config['drivername'].split('+')[0]
    if db_type=='sqlite':
        return f"sqlite:///{config['database']}"
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
        return url
    return URL.create(
        drivername=config['drivername'],
        username=config['username'],
        password=config['password'],
        host=config['host'],
        port=config['port'],
        database=config['database']
    )

def test_database_connection(config):
    engine= None
    try:
        url =create_engine_url(config)
        print(f"\nURL de connexion: {url}")
        engine=create_engine(url)
        with engine.connect() as _:
            print("\nConnected!!!!")
            metadata=MetaData()
            metadata.reflect(bind=engine)
            if metadata.tables:
                print("\nTables disponibles:")
                for table in metadata.sorted_tables:
                    print(f"- {table.name}")
            else:
                print("\nAucune table trouvée dans la base de données.")
            return True
    except OperationalError as e:
        print(f"\n Connection Error: {e}")
        print("Possible Solutions: ")
        if 'mysql' in config ['drivername']:
            print("-verify mySQL")
            print("-Download Driver: pip install pymysql")
            print("-Create DataBase")
        elif 'postgresql' in config ['drivername']:
            print("-verify postgresql")
            print("-Download Driver: pip install psycopg2-binary")
        elif 'oracle' in config ['drivername']:
            print("-verify Oracle")
            print("-Download Driver: pip install cx_Oracle")
            print("-Verify Service Name/SID")
            print("-Download Oracle Client Instances")
        elif 'sqlite' in config ['drivername']:
            print(f"verify file path : {config['database']}")
            print("Make Sure Directory exist")
        return False
    
    except ModuleNotFoundError as e :
        print(f"\n Inexistent Module : {e}")
        print("Download required driver:")
        if 'pymysql' in str(e):
            print("pip install pymysql")
        elif 'psycopg2' in str(e):
            print("pip install psycopg2-binary")
        elif 'cx_Oracle' in str(e):
            print("pip install cx_Oracle")
        elif 'pyodbc' in str(e):
            print("pip install pyodbc")
        return False
    except SQLAlchemyError as e :
        print(f"\n SQLAlchemy Error: {e}")
        return False
    except Exception as e : 
        print(f"\n Unexpected Error : {e}")
        return False
    finally:
        if engine:
            engine.dispose()


def install_drivers():
    print("Downloading Recommanded drivers...")
    os.system("pip install pymysql psycopg2-binary cx_Oracle pyodbc")


if __name__ == "__main__":
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
            print("Operation Cancelled.")
            break