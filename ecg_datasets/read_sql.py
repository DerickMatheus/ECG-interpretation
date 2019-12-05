import psycopg2
from sshtunnel import SSHTunnelForwarder
import numpy as np


def connect(database, sshtunnel=None):
    """Connect to database, possibly through ssh tunnel"""
    if sshtunnel is not None:
        server = SSHTunnelForwarder(
            local_bind_address=('localhost',),
            remote_bind_address=('localhost', int(database['port'])),
            **sshtunnel
        )
        server.start()
        database['host'] = 'localhost'
        database['port'] = str(server.local_bind_port)
    return psycopg2.connect(**database)


def get_ids(conn, table, verbose=False):
    """Get ids from signals table"""
    with conn.cursor() as curs:
        curs.execute('SELECT {2} FROM {0}.{1}'.format(table['schema'],
                                                      table['table'],
                                                      table['id']))
        ids = np.array(curs.fetchall(), dtype=np.int32).flatten()
    if verbose:
        print("There are {0} entries on table ``{1}``".format(len(ids),table['table']))
    return np.unique(ids)  # Remove repeated entries


def get_content(conn, table, id_exam_c):
    # Generate SQL query
    sql = 'SELECT {2}, {3} FROM {0}.{1} S '.format(table['schema'],
                                                   table['table'],
                                                   table['id'],
                                                   table['content'])
    sql += ' WHERE '
    for k, id_n in enumerate(id_exam_c):
        sql += 'S.{0} = {1} '.format(table["id"], id_n)
        if k < len(id_exam_c) - 1:
            sql += ' OR '
    # Send SQL query and get result
    with conn.cursor() as curs:
        curs.execute(sql)
        content_dict = dict(curs.fetchall())
    return content_dict
