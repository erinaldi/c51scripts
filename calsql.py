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
        subprocess.call('ssh -fCN %s@edison.nersc.gov -o TCPKeepAlive=yes -L 5555:scidb1.nersc.gov:5432' %(nerscusr), shell=True)
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
    def insert_mres_v1_meta(self,params,mq):
        fit_id = params['decay_ward_fit']['mres_fit_id']
        ens = params['decay_ward_fit']['ens']['tag']
        m = params['decay_ward_fit'][mq]
        mp_id = params[ens]['mres'][m]['meta_id']['mp']
        pp_id = params[ens]['mres'][m]['meta_id']['pp']
        tmin = params[ens]['mres'][m]['trange']['tmin'][0]
        tmax = params[ens]['mres'][m]['trange']['tmax'][0]
        sqlcmd = "SELECT callat_fcn.upsert($$INSERT INTO callat_proj.mres_v1 (mp_id, pp_id, tmin, tmax, fit_id) VALUES (%s, %s, %s, %s, %s)$$)" %(mp_id, pp_id, tmin, tmax, fit_id)
        self.cur.execute(sqlcmd)
        self.conn.commit()
    def insert_mres_v1_result(self,params,nbs,mbs,result,mq):
        # find meta_id
        fit_id = params['decay_ward_fit']['mres_fit_id']
        ens = params['decay_ward_fit']['ens']['tag']
        m = params['decay_ward_fit'][mq]
        mp_id = params[ens]['mres'][m]['meta_id']['mp']
        pp_id = params[ens]['mres'][m]['meta_id']['pp']
        tmin = params[ens]['mres'][m]['trange']['tmin'][0]
        tmax = params[ens]['mres'][m]['trange']['tmax'][0]
        sqlcmd = "SELECT id FROM callat_proj.mres_v1 WHERE mp_id = %s AND pp_id = %s AND tmin = %s AND tmax = %s AND fit_id = %s;" %(mp_id, pp_id, tmin, tmax, fit_id)
        self.cur.execute(sqlcmd)
        meta_id = str(self.cur.fetchone()[0])
        # find bs_id
        sqlcmd = "SELECT id FROM callat_corr.hisq_ensembles WHERE tag = '%s';" %ens
        self.cur.execute(sqlcmd)
        ens_id = self.cur.fetchone()[0]
        sqlcmd = "SELECT id FROM callat_corr.hisq_bootstrap WHERE hisq_ensembles_id = %s AND nbs = %s;" %(ens_id, nbs)
        self.cur.execute(sqlcmd)
        bs_id = self.cur.fetchone()[0]
        # write result
        result = str(result).replace("'",'\"')
        sqlcmd = "SELECT callat_fcn.upsert($$INSERT INTO callat_proj.mres_v1_bs (mres_v1_id, bs_id, mbs, result) VALUES (%s, %s, %s, '%s')$$)" %(meta_id, bs_id, mbs, result)
        self.cur.execute(sqlcmd)
        self.conn.commit()
    def insert_meson_v1_meta(self,params,meson):
        if meson=='pion':
            mq1 = params['decay_ward_fit']['ml']
            mq2 = mq1
        elif meson == 'kaon':
            mq1 = params['decay_ward_fit']['ml']
            mq2 = params['decay_ward_fit']['ms']
        elif meson == 'etas':
            mq1 = params['decay_ward_fit']['ms']
            mq2 = mq1
        fit_id = params['decay_ward_fit']['meson_fit_id']
        ens = params['decay_ward_fit']['ens']['tag']
        ss_id = params[ens][meson]['%s_%s' %(mq1,mq2)]['meta_id']['SS']
        ps_id = params[ens][meson]['%s_%s' %(mq1,mq2)]['meta_id']['PS']
        tmin = params[ens][meson]['%s_%s' %(mq1,mq2)]['trange']['tmin'][0]
        tmax = params[ens][meson]['%s_%s' %(mq1,mq2)]['trange']['tmax'][0]
        sqlcmd = "SELECT callat_fcn.upsert($$INSERT INTO callat_proj.meson_v1 (meson_ss_id, meson_ps_id, tmin, tmax, fit_id) VALUES (%s, %s, %s, %s, %s)$$)" %(ss_id, ps_id, tmin, tmax, fit_id)
        self.cur.execute(sqlcmd)
        self.conn.commit()
    def insert_meson_v1_result(self,params,nbs,mbs,result,meson):
        if meson=='pion':
            mq1 = params['decay_ward_fit']['ml']
            mq2 = mq1
        elif meson == 'kaon':
            mq1 = params['decay_ward_fit']['ml']
            mq2 = params['decay_ward_fit']['ms']
        elif meson == 'etas':
            mq1 = params['decay_ward_fit']['ms']
            mq2 = mq1
        fit_id = params['decay_ward_fit']['meson_fit_id']
        ens = params['decay_ward_fit']['ens']['tag']
        ss_id = params[ens][meson]['%s_%s' %(mq1,mq2)]['meta_id']['SS']
        ps_id = params[ens][meson]['%s_%s' %(mq1,mq2)]['meta_id']['PS']
        tmin = params[ens][meson]['%s_%s' %(mq1,mq2)]['trange']['tmin'][0]
        tmax = params[ens][meson]['%s_%s' %(mq1,mq2)]['trange']['tmax'][0]
        sqlcmd = "SELECT id FROM callat_proj.meson_v1 WHERE meson_ss_id = %s AND meson_ps_id = %s AND tmin = %s AND tmax = %s AND fit_id = %s;" %(ss_id, ps_id, tmin, tmax, fit_id)
        self.cur.execute(sqlcmd)
        meta_id = str(self.cur.fetchone()[0])
        # find bs_id
        sqlcmd = "SELECT id FROM callat_corr.hisq_ensembles WHERE tag = '%s';" %ens
        self.cur.execute(sqlcmd)
        ens_id = self.cur.fetchone()[0]
        sqlcmd = "SELECT id FROM callat_corr.hisq_bootstrap WHERE hisq_ensembles_id = %s AND nbs = %s;" %(ens_id, nbs)
        self.cur.execute(sqlcmd)
        bs_id = self.cur.fetchone()[0]
        # write result
        result = str(result).replace("'",'\"')
        sqlcmd = "SELECT callat_fcn.upsert($$INSERT INTO callat_proj.meson_v1_bs (meson_v1_id, bs_id, mbs, result) VALUES (%s, %s, %s, '%s')$$)" %(meta_id, bs_id, mbs, result)
        self.cur.execute(sqlcmd)
        self.conn.commit()
    def insert_mixed_v1_meta(self,params,meson):
        tag = params['mixed_fit']['ens']['tag']
        ml = "0.%s" %tag.split('m')[1]
        ms = "0.%s" %tag.split('m')[2]
        if meson in ['phi_ju']:
            mq1 = ml
            mq2 = params['mixed_fit']['ml']
        elif meson in ['phi_js']:
            mq1 = ml
            mq2 = params['mixed_fit']['ms']
        elif meson in ['phi_ru']:
            mq1 = ms
            mq2 = params['mixed_fit']['ml']
        elif meson in ['phi_rs']:
            mq1 = ms
            mq2 = params['mixed_fit']['ms']
        fit_id = params['mixed_fit']['mixed_fit_id']
        ens = params['mixed_fit']['ens']['tag']
        ps_id = params[ens][meson]['%s_%s' %(mq1,mq2)]['meta_id']
        tmin = params[ens][meson]['%s_%s' %(mq1,mq2)]['trange']['tmin'][0]
        tmax = params[ens][meson]['%s_%s' %(mq1,mq2)]['trange']['tmax'][0]
        sqlcmd = "SELECT callat_fcn.upsert($$INSERT INTO callat_proj.mixed_v1 (corr_id, tmin, tmax, fit_id) VALUES (%s, %s, %s, %s)$$)" %(ps_id, tmin, tmax, fit_id)
        self.cur.execute(sqlcmd)
        self.conn.commit()
    def insert_mixed_v1_result(self,params,nbs,mbs,result,meson):
        tag = params['mixed_fit']['ens']['tag']
        ml = "0.%s" %tag.split('m')[1]
        ms = "0.%s" %tag.split('m')[2]
        if meson in ['phi_ju']:
            mq1 = ml
            mq2 = params['mixed_fit']['ml']
        elif meson in ['phi_js']:
            mq1 = ml
            mq2 = params['mixed_fit']['ms']
        elif meson in ['phi_ru']:
            mq1 = ms
            mq2 = params['mixed_fit']['ml']
        elif meson in ['phi_rs']:
            mq1 = ms
            mq2 = params['mixed_fit']['ms']
        fit_id = params['mixed_fit']['mixed_fit_id']
        ens = params['mixed_fit']['ens']['tag']
        ps_id = params[ens][meson]['%s_%s' %(mq1,mq2)]['meta_id']
        tmin = params[ens][meson]['%s_%s' %(mq1,mq2)]['trange']['tmin'][0]
        tmax = params[ens][meson]['%s_%s' %(mq1,mq2)]['trange']['tmax'][0]
        sqlcmd = "SELECT id FROM callat_proj.mixed_v1 WHERE corr_id = %s AND tmin = %s AND tmax = %s AND fit_id = %s;" %(ps_id, tmin, tmax, fit_id)
        self.cur.execute(sqlcmd)
        meta_id = str(self.cur.fetchone()[0])
        # find bs_id
        sqlcmd = "SELECT id FROM callat_corr.hisq_ensembles WHERE tag = '%s';" %ens
        self.cur.execute(sqlcmd)
        ens_id = self.cur.fetchone()[0]
        sqlcmd = "SELECT id FROM callat_corr.hisq_bootstrap WHERE hisq_ensembles_id = %s AND nbs = %s;" %(ens_id, nbs)
        self.cur.execute(sqlcmd)
        bs_id = self.cur.fetchone()[0]
        # write result
        result = str(result).replace("'",'\"')
        sqlcmd = "SELECT callat_fcn.upsert($$INSERT INTO callat_proj.mixed_v1_bs (mixed_v1_id, bs_id, mbs, result) VALUES (%s, %s, %s, '%s')$$)" %(meta_id, bs_id, mbs, result)
        self.cur.execute(sqlcmd)
        self.conn.commit()
    def insert_gA_v1_meta(self,params):
        fit_id = params['gA_fit']['fit_id']
        ens = params['gA_fit']['ens']['tag']
        ml = params['gA_fit']['ml']
        proton_ss_id = params[ens]['proton'][ml]['meta_id']['SS']['G1G1']
        proton_ps_id = params[ens]['proton'][ml]['meta_id']['PS']['G1G1']
        proton_tmin = params[ens]['proton'][ml]['trange']['tmin'][0]
        proton_tmax = params[ens]['proton'][ml]['trange']['tmax'][0]
        axial_ss_id = params[ens]['gA'][ml]['meta_id']['SS']['G1G1']
        axial_ps_id = params[ens]['gA'][ml]['meta_id']['PS']['G1G1']
        axial_tmin = params[ens]['gA'][ml]['trange']['tmin'][0]
        axial_tmax = params[ens]['gA'][ml]['trange']['tmax'][0]
        vector_ss_id = params[ens]['gV'][ml]['meta_id']['SS']['G1G1']
        vector_ps_id = params[ens]['gV'][ml]['meta_id']['PS']['G1G1']
        vector_tmin = params[ens]['gV'][ml]['trange']['tmin'][0]
        vector_tmax = params[ens]['gV'][ml]['trange']['tmax'][0]
        sqlcmd = "SELECT callat_fcn.upsert($$INSERT INTO callat_proj.gA_v1 (proton_ss_id, proton_ps_id, axial_ss_id, axial_ps_id, vector_ss_id, vector_ps_id, proton_tmin, proton_tmax, axial_tmin, axial_tmax, vector_tmin, vector_tmax, fit_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)$$)" %(proton_ss_id, proton_ps_id, axial_ss_id, axial_ps_id, vector_ss_id, vector_ps_id, proton_tmin, proton_tmax, axial_tmin, axial_tmax, vector_tmin, vector_tmax, fit_id)
        self.cur.execute(sqlcmd)
        self.conn.commit()
    def insert_gA_v1_result(self,params,nbs,mbs,result):
        # find meta_id
        fit_id = params['gA_fit']['fit_id']
        ens = params['gA_fit']['ens']['tag']
        ml = params['gA_fit']['ml']
        proton_ss_id = params[ens]['proton'][ml]['meta_id']['SS']['G1G1']
        proton_ps_id = params[ens]['proton'][ml]['meta_id']['PS']['G1G1']
        proton_tmin = params[ens]['proton'][ml]['trange']['tmin'][0]
        proton_tmax = params[ens]['proton'][ml]['trange']['tmax'][0]
        axial_ss_id = params[ens]['gA'][ml]['meta_id']['SS']['G1G1']
        axial_ps_id = params[ens]['gA'][ml]['meta_id']['PS']['G1G1']
        axial_tmin = params[ens]['gA'][ml]['trange']['tmin'][0]
        axial_tmax = params[ens]['gA'][ml]['trange']['tmax'][0]
        vector_ss_id = params[ens]['gV'][ml]['meta_id']['SS']['G1G1']
        vector_ps_id = params[ens]['gV'][ml]['meta_id']['PS']['G1G1']
        vector_tmin = params[ens]['gV'][ml]['trange']['tmin'][0]
        vector_tmax = params[ens]['gV'][ml]['trange']['tmax'][0]
        sqlcmd = "SELECT id FROM callat_proj.gA_v1 WHERE proton_ss_id = %s AND proton_ps_id = %s AND axial_ss_id = %s AND axial_ps_id = %s AND vector_ss_id = %s AND vector_ps_id = %s AND proton_tmin = %s AND proton_tmax = %s AND axial_tmin = %s AND axial_tmax = %s AND vector_tmin = %s AND vector_tmax = %s AND fit_id = %s;" %(proton_ss_id, proton_ps_id, axial_ss_id, axial_ps_id, vector_ss_id, vector_ps_id, proton_tmin, proton_tmax, axial_tmin, axial_tmax, vector_tmin, vector_tmax, fit_id)
        self.cur.execute(sqlcmd)
        meta_id = str(self.cur.fetchone()[0])
        # find bs_id
        sqlcmd = "SELECT id FROM callat_corr.hisq_ensembles WHERE tag = '%s';" %ens
        self.cur.execute(sqlcmd)
        ens_id = self.cur.fetchone()[0]
        sqlcmd = "SELECT id FROM callat_corr.hisq_bootstrap WHERE hisq_ensembles_id = %s AND nbs = %s;" %(ens_id, nbs)
        self.cur.execute(sqlcmd)
        bs_id = self.cur.fetchone()[0]
        # write result
        result = str(result).replace("'",'\"')
        sqlcmd = "SELECT callat_fcn.upsert($$INSERT INTO callat_proj.gA_v1_bs (gA_v1_id, bs_id, mbs, result) VALUES (%s, %s, %s, '%s')$$)" %(meta_id, bs_id, mbs, result)
        self.cur.execute(sqlcmd)
        self.conn.commit()
    def fetch_draws(self,ens,nbs,mbs=False):
        sqlcmd = "SELECT id from callat_corr.hisq_ensembles where tag = '%s';" %ens
        self.cur.execute(sqlcmd)
        meta_id = self.cur.fetchone()[0]
        if mbs in [False, 1.5, 2]:
            sqlcmd = "SELECT draws FROM callat_corr.hisq_bootstrap WHERE nbs='0' AND hisq_ensembles_id='%s';" %meta_id
            self.cur.execute(sqlcmd)
            mbslength = len(self.cur.fetchone()[0])
            if mbs in [1.5, 2]:
                mbslength = mbslength*mbs
        sqlcmd = "SELECT nbs, draws FROM callat_corr.hisq_bootstrap WHERE nbs=0 AND hisq_ensembles_id='%s' UNION SELECT nbs, draws[0:%s] FROM callat_corr.hisq_bootstrap WHERE nbs>=1 AND nbs<=%s AND hisq_ensembles_id='%s' ORDER BY nbs;" %(meta_id,mbslength,nbs,meta_id)
        self.cur.execute(sqlcmd)
        #bs = np.squeeze(np.array(self.cur.fetchall()))
        bs = self.cur.fetchall()
        return bs
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
            sql_cmd = "SELECT * from summary_fhbaryon where meta_id='%s'" %str(meta_id)
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
        init_id = np.sort(init_id).tolist()
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
    def submit_fhboot0(self,tbl,corr_lst,baryon_id,fit_id,tmin,tmax,init,prior,result,update=False):
        # get header
        sql_cmd = "SELECT column_name FROM information_schema.columns WHERE table_name='%s';" %str(tbl)
        self.cur.execute(sql_cmd)
        header = np.squeeze(self.cur.fetchall())[1:-2]
        # get all correlator ids
        ncorr = sum(['fhcorr' in c for c in header])
        corr = [str(int(i)) for i in np.concatenate((np.sort(corr_lst),[0 for c in range(ncorr-len(corr_lst))]))]
        # write values
        values = "'%s','%s','%s','%s','%s','%s','%s','%s'" %("','".join(corr),baryon_id,fit_id,tmin,tmax,init,prior,result)
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
        # get header
        sql_cmd = "SELECT column_name FROM information_schema.columns WHERE table_name='%s';" %str(tbl)
        self.cur.execute(sql_cmd)
        header = np.squeeze(self.cur.fetchall())[1:-2]
        # get all correlator ids
        ncorr = sum(['corr' in c for c in header])
        corr_lst = [str(int(i)) for i in np.concatenate((np.sort(corr_lst),[0 for c in range(ncorr-len(corr_lst))]))]
        # get boot0id
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
