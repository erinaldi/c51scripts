import psycopg2 as psql
import psycopg2.extras as psqle
import numpy as np
import subprocess
import random
import tqdm
try:
    import password_file as pwd
except:
    pass

class pysql():
    def __init__(self,nerscusr,sqlusr,pwd):
        subprocess.call('ssh -fCN %s@cori.nersc.gov -o TCPKeepAlive=yes -L 5555:scidb1.nersc.gov:5432' %(nerscusr), shell=True)
        hostname='localhost'
        databasename='c51_project2'
        portnumber='5555'
        try:
            self.conn = psql.connect(host=hostname, port=portnumber, database=databasename, user=sqlusr, password=pwd)
            self.cur = self.conn.cursor()
            self.dict_cur = self.conn.cursor(cursor_factory=psqle.RealDictCursor)
            self.user = sqlusr
            print "connected"
        except psql.Error as e:
            print "unable to connect to DB"
            print e
            print e.pgcode
            print e.pgerror
            print traceback.format_exc()
            print "exiting"
            raise SystemExit
    def fetchbss(self,Nbs,Mbs,ens=None,stream=None,ens_id=None,bstype='draw'):
        if ens != None and stream != None:
            sql_cmd = "SELECT id FROM callat_corr.hisq_ensembles WHERE tag='%s' AND stream='%s';" %(ens,stream)
            self.cur.execute(sql_cmd)
            meta_id = self.cur.fetchone()[0]
            sql_cmd = "SELECT %s FROM callat_corr.hisq_bootstrap JOIN callat_data.hisq_bootstrap ON callat_corr.hisq_bootstrap.id = callat_data.hisq_bootstrap.meta_id WHERE hisq_ensembles_id='%s' AND nbs>0 AND nbs<='%s' AND mbs<'%s' ORDER BY nbs, mbs;" %(bstype,str(meta_id),str(Nbs),str(Mbs)) 
            self.cur.execute(sql_cmd)
            bss = np.reshape(np.squeeze(np.array(self.cur.fetchall())),(Nbs,Mbs))
            return bss
        elif ens_id != None:
            sql_cmd = "SELECT %s FROM callat_corr.hisq_bootstrap JOIN callat_data.hisq_bootstrap ON callat_corr.hisq_bootstrap.id = callat_data.hisq_bootstrap.meta_id WHERE hisq_ensembles_id='%s' AND nbs>0 AND nbs<='%s' AND mbs<'%s' ORDER BY nbs, mbs;" %(bstype,str(ens_id),str(Nbs),str(Mbs))
            self.cur.execute(sql_cmd)
            bss = np.reshape(np.squeeze(np.array(self.cur.fetchall())),(Nbs,Mbs))
            return bss
        else:
            print "enter either 'ens' and 'stream', or 'meta_id'"
            print "exiting"
            raise SystemExit
    def fetchonebs(self,nbs,Mbs,ens=None,stream=None,ens_id=None,bstype='draw'):
        if ens != None and stream != None:
            sql_cmd = "SELECT id FROM callat_corr.hisq_ensembles WHERE tag='%s' AND stream='%s';" %(ens,stream)
            self.cur.execute(sql_cmd)
            meta_id = self.cur.fetchone()[0]
            sql_cmd = "SELECT %s FROM callat_corr.hisq_bootstrap bs JOIN callat_data.hisq_bootstrap ON bs.id = callat_data.hisq_bootstrap.meta_id WHERE hisq_ensembles_id='%s' AND nbs='%s' AND mbs<'%s' ORDER BY mbs;" %(bstype,str(meta_id),str(nbs),str(Mbs))
            self.cur.execute(sql_cmd)
            bss = np.squeeze(np.array(self.cur.fetchall()))
            sql_cmd = "select bs.id FROM callat_corr.hisq_bootstrap bs JOIN callat_data.hisq_bootstrap ON bs.id = callat_data.hisq_bootstrap.meta_id WHERE hisq_ensembles_id='%s' AND nbs='%s' AND mbs<'%s' ORDER BY mbs;" %(str(meta_id),str(nbs),str(Mbs))
            self.cur.execute(sql_cmd)
            bs_id = self.cur.fetchone()[0]
            return bs_id,bss
        elif ens_id != None:
            sql_cmd = "SELECT %s FROM callat_corr.hisq_bootstrap bs JOIN callat_data.hisq_bootstrap ON bs.id = callat_data.hisq_bootstrap.meta_id WHERE hisq_ensembles_id='%s' AND nbs='%s' AND mbs<'%s' ORDER BY mbs;" %(bstype,str(ens_id),str(nbs),str(Mbs))
            self.cur.execute(sql_cmd)
            bss = np.squeeze(np.array(self.cur.fetchall()))
            sql_cmd = "select bs.id FROM callat_corr.hisq_bootstrap bs JOIN callat_data.hisq_bootstrap ON bs.id = callat_data.hisq_bootstrap.meta_id WHERE hisq_ensembles_id='%s' AND nbs='%s' AND mbs<'%s' ORDER BY mbs;" %(str(meta_id),str(nbs),str(Mbs))
            self.cur.execute(sql_cmd)
            bs_id = self.cur.fetchone()[0]
            return bs_id,bss
        else:
            print "enter either 'ens' and 'stream', or 'meta_id'"
            print "exiting"
            raise SystemExit
    def fetchmeta(self,tbl,meta_id):
        if tbl=='dwhisq_corr_jmu':
            sql_cmd = "SELECT * from summary_jmu where meta_id='%s'" %str(meta_id)
            self.dict_cur.execute(sql_cmd)
            return self.dict_cur.fetchone()
        elif tbl in ['dwhisq_corr_meson','meson_twopt']:
            sql_cmd = "SELECT * from summary_meson where meta_id='%s'" %str(meta_id)
            self.dict_cur.execute(sql_cmd)
            return self.dict_cur.fetchone()
        elif tbl=='dwhisq_corr_baryon':
            sql_cmd = "SELECT * from summary_baryon where meta_id='%s'" %str(meta_id)
            self.dict_cur.execute(sql_cmd)
            return self.dict_cur.fetchone()
        elif tbl=='dwhisq_fhcorr_baryon':
            sql_cmd = "SELECT * from summary_baryon where meta_id='%s'" %str(meta_id)
            self.dict_cur.execute(sql_cmd)
            return self.dict_cur.fetchone()
        else:
            print "table does not exist"
            print "exiting"
            raise SystemExit
    def data(self,tbl,meta_id,nbs=0,mbs=None,trange=None):
        # need to add nbs and mbs calls
        meta = self.fetchmeta(tbl,meta_id)
        if trange==None:
            T = meta['nt']
            sql_cmd = "SELECT rect FROM callat_data.%s WHERE meta_id='%s' ORDER BY config,t" %(tbl,str(meta_id))
            self.cur.execute(sql_cmd)
            return np.reshape(np.squeeze(np.array(self.cur.fetchall())),(meta['cfgs'],T))
        elif trange!=None and nbs==0:
            T = trange[1]-trange[0]+1
            sql_cmd = "SELECT rect FROM callat_data.%s WHERE meta_id='%s' AND t>='%s' AND t<='%s' ORDER BY config,t" %(tbl,str(meta_id),str(trange[0]),str(trange[1]))
            self.cur.execute(sql_cmd)
            return np.reshape(np.squeeze(np.array(self.cur.fetchall())),(meta['cfgs'],T))
    def initid(self,json):
        for k in json.keys():
            sql_cmd = "SELECT callat_fcn.upsert($$INSERT INTO callat_proj.init(tag,mean) VALUES ('%s','%s')$$);" %(str(k),str(json[k]))
            self.cur.execute(sql_cmd)
            self.conn.commit()
        init_id = []
        for k in json.keys():
            sql_cmd = "SELECT id FROM callat_proj.init WHERE tag='%s' AND mean='%s';" %(str(k),str(json[k]))
            self.cur.execute(sql_cmd)
            init_id.append(self.cur.fetchone()[0])
        init_id = str(init_id).replace('[','{').replace(']','}')
        return init_id
    def priorid(self,json):
        for k in json.keys():
            sql_cmd = "SELECT callat_fcn.upsert($$INSERT INTO callat_proj.prior(tag,mean,sdev) VALUES ('%s','%s','%s')$$);" %(str(k),str(json[k].mean),str(json[k].sdev))
            self.cur.execute(sql_cmd)
            self.conn.commit()
        prior_id = []
        for k in json.keys():
            sql_cmd = "SELECT id FROM callat_proj.prior WHERE tag='%s' AND mean='%s' AND sdev='%s';" %(str(k),str(json[k].mean),str(json[k].sdev))
            self.cur.execute(sql_cmd)
            prior_id.append(self.cur.fetchone()[0])
        prior_id = np.sort(prior_id).tolist()
        prior_id = str(prior_id).replace('[','{').replace(']','}')
        return prior_id
    def bspriorid(self,tag,stream,g,json):
        # get ensemble meta_id
        sql_cmd = "SELECT id FROM callat_corr.hisq_ensembles WHERE tag='%s' AND stream='%s';" %(str(tag),str(stream))
        self.cur.execute(sql_cmd)
        ens_id = self.cur.fetchone()[0]
        # get bs_id
        sql_cmd = "SELECT id FROM callat_corr.hisq_bootstrap WHERE hisq_ensembles_id='%s' AND nbs='%s';" %(str(ens_id),str(g))
        self.cur.execute(sql_cmd)
        bs_id = self.cur.fetchone()[0]
        # get prior_ids
        bsdict = dict()
        bslst = []
        for k in json.keys():
            sql_cmd = "SELECT id FROM callat_proj.prior WHERE tag='%s' AND mean='%s' AND sdev='%s';" %(str(k),str(json[k].mean),str(json[k].sdev))
            self.cur.execute(sql_cmd)
            prior_id = self.cur.fetchone()[0]
            sql_cmd = "SELECT id,mean,sdev FROM callat_proj.bsprior WHERE prior_id='%s' AND bs_id='%s';" %(str(prior_id),str(bs_id))
            self.cur.execute(sql_cmd)
            query = self.cur.fetchall()[0]
            bslst.append(query[0])
            bsdict[k] = [query[1],query[2]]
        bslst = np.sort(bslst).tolist()
        bsprior_id = str(bslst).replace('[','{').replace(']','}')
        return bsprior_id,bsdict
    def chkbsprior(self,tag,stream,Nbs,json):
        # get ensemble meta_id
        sql_cmd = "SELECT id FROM callat_corr.hisq_ensembles WHERE tag='%s' AND stream='%s';" %(str(tag),str(stream))
        self.cur.execute(sql_cmd)
        ens_id = self.cur.fetchone()[0]
        # select boot0 prior keys
        prior_id = []
        for k in json.keys():
            sql_cmd = "SELECT id FROM callat_proj.prior WHERE tag='%s' AND mean='%s' AND sdev='%s';" %(str(k),str(json[k].mean),str(json[k].sdev))
            self.cur.execute(sql_cmd)
            prior_id.append(self.cur.fetchone()[0])
        n = dict()
        for p in prior_id:
            sql_cmd = "SELECT count(*) FROM callat_proj.bsprior p JOIN callat_corr.hisq_bootstrap bs ON p.bs_id=bs.id JOIN callat_corr.hisq_ensembles e ON bs.hisq_ensembles_id=e.id WHERE p.prior_id='%s' AND e.tag='%s' AND e.stream='%s';" %(str(p),str(tag),str(stream))
            self.cur.execute(sql_cmd)
            n['%s' %str(p)] = self.cur.fetchone()[0]
        for k in n.keys():
            if n[k] < Nbs:
                sql_cmd = "SELECT tag,mean,sdev FROM callat_proj.prior WHERE id='%s';" %str(k)
                self.cur.execute(sql_cmd)
                p = self.cur.fetchall()[0]
                tag = p[0]
                print "Need to generate bootstrap list for prior %s" %str(tag)
                flag = 'yes' #raw_input("Generate bootstrap. Redraw with %s=%s+-%s? (yes/nonononono)\n"%(str(tag),str(p[1]),str(p[2])))
                if flag=='yes':
                    for g in tqdm.tqdm(range(2000)):
                        sql_cmd = "SELECT id FROM callat_corr.hisq_bootstrap WHERE hisq_ensembles_id='%s' AND nbs='%s';" %(str(ens_id),str(g))
                        self.cur.execute(sql_cmd)
                        bs_id = self.cur.fetchone()[0]
                        bsmean = random.gauss(p[1],p[2])
                        sql_cmd = "SELECT callat_fcn.upsert($$INSERT INTO callat_proj.bsprior(prior_id,bs_id,mean,sdev) VALUES ('%s','%s','%s','%s')$$);" %(str(k),str(bs_id),str(bsmean),str(p[2]))
                        self.cur.execute(sql_cmd)
                        self.conn.commit()
                else:
                    print "why is this even an option?"
                    print "exiting program"
                    raise SystemExit
    def submit_boot0(self,tbl,corr_lst,fit_id,tmin,tmax,init,prior,result,update=False):
        # get header
        sql_cmd = "SELECT column_name FROM information_schema.columns WHERE table_name='%s';" %str(tbl)
        self.cur.execute(sql_cmd)
        header = np.squeeze(self.cur.fetchall())[1:-2]
        # get all correlator ids
        ncorr = sum(['corr' in c for c in header])
        corr = [str(int(i)) for i in np.concatenate((np.sort(corr_lst),[0 for c in range(ncorr-len(corr_lst))]))]
        # write values
        values = "'%s','%s','%s','%s','%s','%s','%s'" %("','".join(corr),fit_id,tmin,tmax,init,prior,result)
        # get header
        header = ','.join(header)
        insert = "INSERT INTO callat_proj.%s (%s) VALUES (%s)" %(tbl,header,values)
        if not update:
            sql_cmd = "SELECT callat_fcn.upsert($$%s$$);" %(insert)
            self.cur.execute(sql_cmd)
            self.conn.commit()
        elif update:
            where = ' AND '.join(["%s=%s'" %(header.split(',')[i],values.split("',")[i]) for i in range(len(header.split(','))-1)]+["commit_user='%s'" %(self.user)])
            update = "UPDATE callat_proj.%s SET (%s)=(%s) WHERE %s" %(tbl,header,values,where)
            sql_cmd = "SELECT callat_fcn.upsert($$%s$$,$$%s$$);" %(insert,update)
            self.cur.execute(sql_cmd)
            self.conn.commit()
        return 0
    def select_boot0(self,tbl,corr_lst,fit_id,tmin,tmax,init_id,prior_id):
        corr = ' AND '.join(["corr%s_id='%s'" %(str(i+1),corr_lst[i]) for i in range(len(corr_lst))])
        sql_cmd = "SELECT id FROM %s WHERE %s AND fit_id='%s' AND tmin='%s' AND tmax='%s' AND init_id='%s' AND prior_id='%s' AND commit_user='%s';" %(str(tbl),str(corr),str(fit_id),str(tmin),str(tmax),init_id,prior_id,str(self.user))
        self.cur.execute(sql_cmd)
        return self.cur.fetchone()[0]
    def submit_bs(self,tbl,boot0_id,bs_id,Mbs,init,prior,result,update=False):
        # get header
        sql_cmd = "SELECT column_name FROM information_schema.columns WHERE table_name='%s';" %str(tbl)
        self.cur.execute(sql_cmd)
        header = np.squeeze(self.cur.fetchall())[:-2]
        # write values
        values = "'%s','%s','%s','%s','%s','%s'" %(str(boot0_id),str(bs_id),str(Mbs),init,prior,result)
        # get header
        header = ','.join(header)
        insert = "INSERT INTO callat_proj.%s (%s) VALUES (%s)" %(tbl,header,values)
        if not update:
            sql_cmd = "SELECT callat_fcn.upsert($$%s$$);" %(insert)
            self.cur.execute(sql_cmd)
            self.conn.commit()
        elif update:
            where = ' AND '.join(["%s=%s'" %(header.split(',')[i],values.split("',")[i]) for i in range(len(header.split(','))-2)]+["commit_user='%s'" %(self.user)])
            update = "UPDATE callat_proj.%s SET (%s)=(%s) WHERE %s" %(tbl,header,values,where)
            sql_cmd = "SELECT callat_fcn.upsert($$%s$$,$$%s$$);" %(insert,update)
            self.cur.execute(sql_cmd)
            self.conn.commit()
        return 0
if __name__=="__main__":
    # examples
    # instantiate first
    psqlpwd = pwd.passwd()
    psql = pysql('cchang5','cchang5',psqlpwd)
    # get data
    if False:
        psql.data('dwhisq_corr_meson',meta_id='1')
    # get data with time range
    if False:
        psql.data('dwhisq_corr_meson',meta_id='1',trange=[0,2])
    # get bootstrap draws with ensemble_id
    if False:
        psql.fetchbss(Nbs=5,Mbs=10,ens_id='0')
    # get bootstrap draws with ensemble tag + stream
    if False:
        psql.fetchbss(5,10,ens='l1648f211b580m013m065m838',stream='a')
    # submit fit result with Mbs = config number
    # corr_lst = [2,1] : simultaneous fit to corrlator_id 2 and 1. Order doesn't matter.
    if False:
        psql.submission(tbl='meson_twopt',corr_lst=[2,1],nbs=0,fit_id=2,tmin=4,tmax=15,input_params="""{"E0":"1.0", "Z0_p":"0.5", "Z0_s":"0.5"}""",result="""{"E0":"1.0", "dE0":"0.4", "Z0":"0.2", "dZ0":"0.1"}""")
    # Mbs = 4000 : submit fit result with given Mbs
    # corr_lst = [1] : single fit to correlator_id 1.
    # update = True : updates duplicate entry instead of skipping
    if True:
        psql.submission(tbl='meson_twopt',corr_lst=[1],nbs=0,Mbs=4000,fit_id=2,tmin=4,tmax=15,input_params="""{"E0":"1.0", "Z0_p":"0.5", "Z0_s":"0.5"}""",result="""{"E0":"10.0", "dE0":"0.4", "Z0":"0.2", "dZ0":"0.1"}""",update=True)
