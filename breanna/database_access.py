import sqlite3
import pandas as pd
import os
from pandas.io.common import EmptyDataError
from breanna import feature_extraction as fe
from breanna.util import to_ospath, copyfromto

DB_PATH = os.path.join(os.path.dirname(__file__),
                       to_ospath('../breanna_data/database/msl_dev.db'))
def get_connection(db_path=DB_PATH):
    return sqlite3.connect(db_path)

######## for creating tables ########

def initialize_database(conn):
    sql_create_tables = [
        """
        create table monthly_campaigns
        (
        id                   integer primary key autoincrement,
        ontology             text not null,
        name                 text not null,
        month                integer not null,
        year                 integer not null,
        banner_root          text not null
        )
        """,
        """
        create table exposures
        (
        id                   integer primary key autoincrement,
        ad_name              text not null,
        publisher            text not null,
        os                   text not null,
        device               text not null,
        event_date           text not null,
        event_hour           integer not null,
        click                integer not null,
        monthly_campaign_id  integer,
        foreign key(monthly_campaign_id) references monthly_campaign(id)
        )
        """
    ]
    
    for sql in sql_create_tables:
        conn.execute(sql)
    conn.commit()
    
######## for inserting records for new campaigns ########
    
def load_new_campaign(ontology, name, month, year, banner_root, 
                      standarddisplay_root, matchfile_root, 
                      conn):
    monthly_campaign_id = add_campaign(ontology, name, month, year, banner_root, conn)
    insert_exposures(standarddisplay_root, matchfile_root, monthly_campaign_id,  conn)

WHERE_BREANNA_KEEPS_BANNERS = os.path.join(os.path.dirname(__file__),
                                           to_ospath('../breanna_data/banners'))
def add_campaign(ontology, name, month, year, banner_root, conn):
    print('Importing banners...')
    if banner_root[-1] == '/':
        base = os.path.basename(banner_root[:-1])
    else:
        base = os.path.basename(banner_root)
    banner_root_breanna = os.path.join(WHERE_BREANNA_KEEPS_BANNERS, base) 
    # import the banners by copying it to the breanna_data/banners/ folder
    copyfromto(banner_root, banner_root_breanna) 
    print('Banners imported...')
    # add '/' banner_root_breanna indicating that it is an absolute path
    new_campaign = (ontology, name, month, year, '/'+str(banner_root_breanna))
    for campaign in conn.execute("select ontology, name, month, year from monthly_campaigns"):
        if campaign == new_campaign:
            raise RuntimeError('duplicate campaign')
    cursor = conn.cursor()
    cursor.execute(
    "insert into monthly_campaigns values (NULL, ?, ?, ?, ?, ?)", new_campaign
    )
    conn.commit()
    return cursor.lastrowid

def insert_exposures(standarddisplay_root, matchfile_root, monthly_campaign_id, conn):
    print('Processing standard display feeds...')
    standarddisplay_root = to_ospath(standarddisplay_root)
    matchfile_root       = to_ospath(matchfile_root)
    standarddisplay_path_list = [
        os.path.join(standarddisplay_root, name) for name in os.listdir(standarddisplay_root)
    ]
    for i, standarddisplay_path in enumerate(standarddisplay_path_list):
        exposures_df = create_dataframe(standarddisplay_path, matchfile_root)
        exposures_df['monthly_campaign_id'] = monthly_campaign_id
        exposures_df.to_sql('exposures', conn, index=False, if_exists='append')
        print(f'{i:4d}/{len(standarddisplay_path_list):4d}')
        
# mappings used in create_dataframe() and create_matchdict()
type2substr = {'ad_name': 'AdsMatchfile', 'publisher': 'SitesMatchfile', 'os': 'OSTypeMatchfile'}
type2keycol = {'ad_name': 'AdID',         'publisher': 'SiteID',         'os': 'OSID'           }
type2valcol = {'ad_name': 'AdName',       'publisher': 'SiteName',       'os': 'OSName'         }
os2device = {
    'Windows':  'Desktop',
    'WINDOWS':  'Desktop',
    'MAC':      'Desktop', 
    'Mac':      'Desktop',
    'Ubuntu':   'Desktop',
    'Linux':    'Desktop',
    'Chrome OS':'Desktop',
    'Android Mobile': 'Android Mobile',
    'android mobile': 'Android Mobile',
    'Android Generic':'Android Mobile',
    'android tablet': 'Android Mobile',
    'Android Table':  'Android Mobile',
    'Android Tablet': 'Android Mobile',
    'iphone':'Apple Mobile',
    'ipad':  'Apple Mobile', 
    'windows phone os':'Other',
    'NOTSUPPORTED':    'Other',
    'Smart TV':        'Other',
    'OTT':             'Other'
}

# subroutines
def create_dataframe(standarddisplay_path, matchfile_root):
    try:
        standarddisplay_df = pd.read_csv(standarddisplay_path, 
                                     delimiter='\x7f', encoding='latin1', header=None)
        timeoffeed = standarddisplay_path.split('_')[-1]
        ad_name_matchdict   = create_matchdict('ad_name',   matchfile_root, timeoffeed)
        publisher_matchdict = create_matchdict('publisher', matchfile_root, timeoffeed)
        os_matchdict        = create_matchdict('os',        matchfile_root, timeoffeed)
    except EmptyDataError:
        # empty standard display feeds seem to be normal, no need to make a fuss about it
        return pd.DataFrame()
    except Exception as e:
        print('empty dataframe returned while processing the following file:')
        print(standarddisplay_path)
        print('error message:', str(e))
        return pd.DataFrame()
    
    return_df = pd.DataFrame()
    return_df['id']         = 0 # it will be replaced with autoincremented PK when inserted in DB
    return_df['ad_name']    = standarddisplay_df[4].map( ad_name_matchdict)
    return_df['publisher']  = standarddisplay_df[6].map( publisher_matchdict)
    return_df['os']         = standarddisplay_df[20].map(os_matchdict)
    return_df['device']     = return_df['os'].map(os2device)
    return_df['event_date'] = standarddisplay_df[3]
    return_df['event_hour'] = return_df['event_date'].apply( lambda dtstr: int(dtstr[11:13]) )
    return_df['click']      = standarddisplay_df[2]-1
    
    # drop non-display event (e.g. search, tracking pixel)
    return_df = return_df[~return_df['ad_name'].isnull() & (return_df['ad_name']!='Tracking Pixel')]
    
    return return_df

def create_matchdict(matchfiletype, matchfile_root, timeoffeed):
    matchfile_name = [name for name in 
                        filenames_with_substr(type2substr[matchfiletype], matchfile_root)
                        if name.endswith(timeoffeed)][0]
    matchfile_path = os.path.join(matchfile_root, matchfile_name)
    matchfile_df   = pd.read_csv(matchfile_path, delimiter='\x7f')
    keycol = type2keycol[matchfiletype]
    valcol = type2valcol[matchfiletype]
    matchfile_dict = dict(zip(matchfile_df[keycol], matchfile_df[valcol]))
    return matchfile_dict

def filenames_with_substr(substr, rootdir):
    return [name for name in os.listdir(rootdir) if substr in name]

######## for querying ########
    
def get_monthly_campaign_ids(ontology, name, year, month, conn):
    ontology, name, year, month = to_sqllist(ontology), to_sqllist(name), to_sqllist(year), to_sqllist(month)
    sql = f"""
    select id from monthly_campaigns
    where ontology in {ontology}
      and name     in {name}
      and year     in {year}
      and month    in {month}
    """
    return [i for i, in conn.execute(sql).fetchall()]

def to_sqllist(pythonlist, used_in='in'):
    if used_in=='in':
        sqllist = tuple(pythonlist)
        if len(sqllist)==1: sqllist = str(sqllist)[:-2]+str(sqllist)[-1]  # remove the last ','
    if used_in=='group by':
        sqllist = ', '.join(pythonlist)
    return sqllist

def aggregate_ctr(monthly_campaign_ids, ctr_aggregators, min_examples, conn):
    monthly_campaign_ids = to_sqllist(monthly_campaign_ids)
    groups = ['ad_name']; groups.extend(ctr_aggregators)
    groups = to_sqllist(groups, used_in='group by')
    ctr_aggregators = to_sqllist(ctr_aggregators, used_in='group by')
    sql = f"""
    select ad_name, {ctr_aggregators}, ctr from
    (
    select ad_name, {ctr_aggregators}, avg(click) as ctr, count(*) as count from exposures
    where monthly_campaign_id in {monthly_campaign_ids} 
    group by {groups}
    )
    where count > {min_examples}
    """
    return pd.read_sql(sql, conn)

def get_banner_roots(ontology, name, year, month, conn):
    ontology, name, year, month = to_sqllist(ontology), to_sqllist(name), to_sqllist(year), to_sqllist(month)
    sql = f"""
    select banner_root from monthly_campaigns
    where ontology in {ontology}
      and name     in {name}
      and year     in {year}
      and month    in {month}
    """
    return [br for br, in conn.execute(sql).fetchall()]

def aggregate_viz(banner_roots, viz_aggregator):
    viz_agg = pd.DataFrame()
    for banner_root in banner_roots:
        viz_agg = viz_agg.append( fe.summarize_banners(banner_root, viz_aggregator) )
    return viz_agg