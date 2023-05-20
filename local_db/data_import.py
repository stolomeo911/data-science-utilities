import pandas as pd
import psycopg2
import os
import csv
from sqlalchemy import create_engine
from sqlalchemy import URL


def read_file(file):
    if file.endswith('.csv'):
        return pd.read_csv(file)
    elif file.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        raise NotImplementedError


def prepare_dateset_for_classifier_model(df, target_column):
    df["target"] = target_column
    return df


def connect_local_db():
    # Define the database connection parameters
    host = os.environ["LOCAL_HOST"]
    dbname = os.environ["DBNAME"]
    user = os.environ["USER_LOCAL_HOST"]
    password = os.environ["PASSWORD_LOCAL_HOST"]
    conn = psycopg2.connect(host=host, dbname=dbname, user=user, password=password)
    conn.autocommit = True
    return conn


def connect_sqlalchemy():
    # Define the database connection parameters
    host = os.environ["LOCAL_HOST"]
    dbname = os.environ["DBNAME"]
    user = os.environ["USER_LOCAL_HOST"]
    password = os.environ["PASSWORD_LOCAL_HOST"]
    url = f"postgresql://{user}:{password}@{host}:5432/{dbname}"
    conn = create_engine(url)
    return url


def create_database_from_csv(files_path, schema):

    conn = connect_local_db()
    schema = schema
    cur = conn.cursor()
    create_schema = f"CREATE SCHEMA IF NOT EXISTS {schema};"
    cur.execute(create_schema)

    # Create the database if it does not already exist
    #conn = psycopg2.connect(host=host, user=user, password=password)
    #conn.autocommit = True
    #cur = conn.cursor()
    #cur.execute(f"CREATE DATABASE {dbname}")
    #cur.close()
    #conn.close()

    # Connect to the newly created database


    # Iterate over each CSV file in the current directory
    for file in os.listdir(files_path):
        if file.endswith(".csv"):
            table_name = file.split(".")[0]  # Extract the table name from the file name
            with open(file, "r") as f:
                reader = csv.reader(f, delimiter=',')
                header = next(reader)
                # Generate the CREATE TABLE statement using the first row of the CSV as column names
                create_table_sql = f"CREATE TABLE IF NOT EXISTS {schema}.{table_name} ({', '.join([f'{col} VARCHAR(200)' for col in header])});"
                print(create_table_sql)
                cur.execute(create_table_sql)
                # Insert the remaining rows into the table
                insert_sql = f"INSERT INTO {schema}.{table_name} ({', '.join(header)}) VALUES ({', '.join(['%s' for _ in header])})"
                cur.executemany(insert_sql, reader)
                cur.close()

    conn.close()


def sql_to_df(sql_file):
    fd = open(sql_file, 'r')
    sql_file = fd.read()
    fd.close()

    conn_database = connect_local_db()
    df = pd.read_sql(sql_file, conn_database)
    return df

