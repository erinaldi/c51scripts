import sys
sys.path.append('$HOME/c51/scripts/')
import calsql as sql
import password_file_admin as pwd
import numpy as np

if __name__=='__main__':
    # log in sql
    tbl0 = 'callat_data.dwhisq_corr_baryon'
    id0 = '270'
    tbl1 = 'callat_data.dwhisq_fhcorr_baryon'
    id1 = '591'

    psqlpwd = pwd.passwd()
    psql = sql.pysql('cchang5','c51_project2_admin',psqlpwd)
    cmd = 'select distinct config from %s where meta_id=%s order by config;' %(tbl0, id0)
    psql.cur.execute(cmd)
    cfg0 = np.array(psql.cur.fetchall()).flatten()
    cmd = 'select distinct config from %s where meta_id=%s order by config;' %(tbl1, id1)
    psql.cur.execute(cmd)
    cfg1 = np.array(psql.cur.fetchall()).flatten()
    if len(tbl0.split('_')[2]) == 6:
        header = 'fh'
    else:
        header = ''
    cmd = 'select ensemble from summary_%s%s where meta_id=%s;' %(header, tbl0.split('_')[3], id0)
    psql.cur.execute(cmd)
    ens = psql.cur.fetchone()[0]
    cmd = "select meta_id from summary_%s%s where ensemble='%s';" %(header, tbl0.split('_')[3], ens)
    psql.cur.execute(cmd)
    meta0 = np.array(psql.cur.fetchall()).flatten()
    if len(tbl1.split('_')[2]) == 6:
        header = 'fh'
    else:
        header = ''
    cmd = "select meta_id from summary_%s%s where ensemble='%s';" %(header, tbl1.split('_')[3], ens)
    psql.cur.execute(cmd)
    meta1 = np.array(psql.cur.fetchall()).flatten()
    for i in cfg0:
        if i not in cfg1:
            print 'deleting from cfg0:', i
            for j in meta0:
                cmd = "delete from %s where meta_id = %s and config = %s;" %(tbl0, j, i)
                psql.cur.execute(cmd)
                psql.conn.commit()
    for i in cfg1:
        if i not in cfg0:
            print 'deleting from cfg1:', i
            for j in meta1:
                cmd = "delete from %s where meta_id = %s and config = %s;" %(tbl1, j, i)
                psql.cur.execute(cmd)
                psql.conn.commit()
