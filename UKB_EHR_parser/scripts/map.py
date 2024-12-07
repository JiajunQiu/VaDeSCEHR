import os
import datetime
import pandas as pd
import numpy as np
from itertools import count
from collections import defaultdict
import re
import pickle
from tqdm import tqdm
chapter={}
for l in open('ICD10_chapter.txt'):
    l=l.rstrip()
    l=l.replace(' ', '\t', 1)
    code,des = l.split('\t')
    c1,c2=code.split('-')
    for x in range(int(c1[1:]),int(c2[1:])+1):
        if x < 10:
            x=c1[0]+'0'+str(x)
        else:
            x=c1[0]+str(x)
        chapter[x]=code

'''
anno_dict = pd.read_excel('./primary_data/all_lkps_maps_v3.xlsx', sheet_name=None)

icd9_icd10=pd.DataFrame(anno_dict["icd9_icd10"])
icd9_icd10.columns = ['ori_code','ori_des','icd10','icd10_des']

read_v2_icd10=pd.DataFrame(anno_dict["read_v2_icd10"])
read_v2_icd10.columns = ['ori_code','icd10','icd10_def']

read_ctv3_icd10=pd.DataFrame(anno_dict["read_ctv3_icd10"])

out=open('./primary_data/read_ctv3_icd10.pkl','rb')
read_v3=pickle.load(out)
out.close()

anno_dict = {'ICD9':icd9_icd10,'read_v2':read_v2_icd10,'read_v3':read_v3}
'''
out=open('./primary_data/anno_dict.pkl','rb')
anno_dict=pickle.load(out)
out.close()

data_code={}
data_type={}
data_time={}
data_age={}

for l in open('./mergered_data/UKB_EHR_diagnosis_time.tsv'):
    t=l.rstrip().split('\t')
    if t[0]=='PID':
        continue
    start=datetime.datetime.strptime(t[1],"%Y-%m-%d")
    end=datetime.datetime.strptime(t[-1],"%Y-%m-%d")
    check = end-start
    if check.days > 30:
        data_time[t[0]]=t[1:]

for l in open('./mergered_data/UKB_EHR_diagnosis_codetype.tsv'):
    t=l.rstrip().split('\t')
    if t[0]=='PID':
        continue
    if t[0] in data_time:
        data_type[t[0]]=t[1:]

for l in open('./mergered_data/UKB_EHR_diagnosis_code.tsv'):
    t=l.rstrip().split('\t')
    if t[0]=='PID':
        continue
    if t[0] in data_time:
        data_code[t[0]]=t[1:]

for l in open('./mergered_data/UKB_EHR_diagnosis_age.tsv'):
    t=l.rstrip().split('\t')
    if t[0]=='PID':
        continue
    if t[0] in data_time:
        data_age[t[0]]=t[1:]

data_code3_cv={}
data_code2_cv={}
data_code1_cv={}
data_type_cv={}
data_time_cv={}
data_age_cv={}
data_multi_cv={}

max_len=0
print('mapping to ICD10 code')
for pid in tqdm(data_code):

    tmp_code3=[]
    tmp_code2=[]
    tmp_code1=[]
    tmp_time=[]
    tmp_type=[]
    tmp_age=[]
    tmp_multi=[]
    multi=0
    for idx,code in enumerate(data_code[pid]):
        tp = data_type[pid][idx]
        if tp=='ICD10':
            if len(code)>3:
                tmp_code1.append(chapter[code[:3]])
                tmp_code2.append(code[:3])
                tmp_code3.append(code)
            else:
                tmp_code1.append(chapter[code])
                tmp_code2.append(code)
                tmp_code3.append('UNK')                
            tmp_time.append(data_time[pid][idx])
            tmp_type.append('1')
            tmp_age.append(data_age[pid][idx])
            tmp_multi.append(str(multi))
        else:
            mapping = anno_dict[tp]['icd10'][anno_dict[tp]['ori_code']==code]
            if 'read' in tp:
                tp_='0'
            else:
                tp_='1'
            if len(mapping)>0:
                for mpcs in mapping:
                    if '-' in mpcs:
                        c1,c2 = mpcs.split('-')
                        check_d = sum(c.isdigit() for c in c1)
                        if check_d>2:
                            mpc=c1[:3]
                            tmp_code1.append(chapter[mpc])
                            tmp_code2.append(mpc)
                            tmp_code3.append('UNK')
                        else:
                            mpc=c1[:3]+'-'+c2[:3]
                            tmp_code1.append(mpc)
                            tmp_code2.append('UNK')
                            tmp_code3.append('UNK')
                        tmp_time.append(data_time[pid][idx])
                        tmp_type.append(tp_)
                        tmp_age.append(data_age[pid][idx])      
                        tmp_multi.append(str(multi))
                    else:
                        mpcs=re.split('\+|\s|,', mpcs)
                        for mpc in mpcs:
                            if mpc=='UNDEF':
                                tmp_code1.append('UNK')
                                tmp_code2.append('UNK')
                                tmp_code3.append(tp+'_'+code)  
                                tmp_time.append(data_time[pid][idx])
                                tmp_type.append(tp_)
                                tmp_age.append(data_age[pid][idx])      
                                tmp_multi.append(str(multi))
                                continue
                            if len(mpc)>3:
                                tmp_code1.append(chapter[mpc[:3]])
                                tmp_code2.append(mpc[:3])
                                tmp_code3.append(mpc)
                            else:
                                tmp_code1.append(chapter[mpc])
                                tmp_code2.append(mpc)
                                tmp_code3.append('UNK')                
                            tmp_time.append(data_time[pid][idx])
                            tmp_type.append(tp_)
                            tmp_age.append(data_age[pid][idx])
                            tmp_multi.append(str(multi))
            else:
                tmp_code1.append('UNK')
                tmp_code2.append('UNK')
                tmp_code3.append(tp+'_'+code)  
                tmp_time.append(data_time[pid][idx])
                tmp_type.append(tp_)
                tmp_age.append(data_age[pid][idx])      
                tmp_multi.append(str(multi))
        multi+=1      
    data_code1_cv[pid]=tmp_code1
    data_code2_cv[pid]=tmp_code2
    data_code3_cv[pid]=tmp_code3
    data_type_cv[pid]=tmp_type
    data_time_cv[pid]=tmp_time
    data_age_cv[pid]=tmp_age
    data_multi_cv[pid]=tmp_multi

out11=open('./output/UKB_EHR_diagnosis_code1.tsv','w')
out12=open('./output/UKB_EHR_diagnosis_code2.tsv','w')
out13=open('./output/UKB_EHR_diagnosis_code3.tsv','w')
out2=open('./output/UKB_EHR_diagnosis_type.tsv','w')
out3=open('./output/UKB_EHR_diagnosis_time.tsv','w')
out4=open('./output/UKB_EHR_diagnosis_age.tsv','w')

out5=open('./output/UKB_EHR_diagnosis_multi.tsv','w')
out6=open('./output/UKB_EHR_diagnosis_segment.tsv','w')


PIDs=list(data_code1_cv.keys())       
for idx,pid in enumerate(PIDs):
    print('\t'.join([pid]+data_code1_cv[pid]),file=out11)
    print('\t'.join([pid]+data_code2_cv[pid]),file=out12)
    print('\t'.join([pid]+data_code3_cv[pid]),file=out13)
    print('\t'.join([pid]+data_type_cv[pid]),file=out2)
    print('\t'.join([pid]+data_time_cv[pid]),file=out3)
    print('\t'.join([pid]+data_age_cv[pid]),file=out4)
    print('\t'.join([pid]+data_multi_cv[pid]),file=out5)
    d = defaultdict(count(0).__next__)
    seg=[str(d[k]) for k in data_time_cv[pid]]
    print('\t'.join([pid]+seg),file=out6)

out11.close()
out12.close()
out13.close()
out2.close()
out3.close()
out4.close()
out5.close()
out6.close()


