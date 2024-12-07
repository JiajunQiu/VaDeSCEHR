import os
import sys
from tqdm import tqdm
#where are your files
main_file=sys.argv[1]

#where you want to save the ouputs
out1f='./primary_data/temp/UKB_EHR_diagnosis_codetype.tsv'
out2f='./primary_data/temp/UKB_EHR_diagnosis_time.tsv'
out3f='./primary_data/temp/UKB_EHR_diagnosis_readcode.tsv'

out1=open(out1f,'w')
out2=open(out2f,'w')
out3=open(out3f,'w')

def sort_by_time(a,b):
    zipped = zip(a, b)
    sorted_zipped = sorted(zipped, key=lambda x: x[1].split('-'))
    a, b = zip(*sorted_zipped)
    return list(a)

readv2_to_ICD9={}
for l in open('./primary_data/read_v2_to_ICD9.txt'):
    t=l.rstrip().split('\t')
    read=t[0]
    icd=t[1]
    if '+' in icd:
        icd=icd.split('+')
    elif '-' in icd:
        icd=icd.split('-')
    elif ' ' in icd:
        icd=icd.split(' ')
    else:
        icd=[icd]
    readv2_to_ICD9[read]=icd


readv3_to_ICD9={}
for l in open('./primary_data/read_v3_to_ICD9.txt'):
    t=l.rstrip().split('\t')
    read, icd, mapping_status, refine_flag, add_code_flag, element_num, block_num = t
    if read not in readv3_to_ICD9:
        readv3_to_ICD9[read]=[]
    if mapping_status in 'EGD' and block_num in '0':
        readv3_to_ICD9[read].append(icd)


readv2_to_ICD10={}
for l in open('./primary_data/read_v2_to_ICD10.txt'):
    t=l.rstrip().split('\t')
    read=t[0]
    icd=t[1]
    if '+' in icd:
        icd=icd.split('+')
    elif '-' in icd:
        icd=icd.split('-')
    elif ' ' in icd:
        icd=icd.split(' ')
    else:
        icd=[icd]
    readv2_to_ICD10[read]=icd


readv3_to_ICD10={}
for l in open('./primary_data/read_v3_to_ICD10.txt'):
    t=l.rstrip().split('\t')
    read, icd, mapping_status, refine_flag, add_code_flag, element_num, block_num = t
    if read not in readv3_to_ICD10:
        readv3_to_ICD10[read]=[]
    if mapping_status in 'EGD' and block_num in '0':
        readv3_to_ICD10[read].append(icd)



data_read={}
data_icd={}
data_time={}
data_type={}
data_check={}

print('parsing primary data')
for l in tqdm(open(main_file, encoding="utf8", errors='ignore')):
    t=l.rstrip().split('\t')
    if t[0] == 'eid' or t[2]=='':
        continue
    pid=t[0]
    if pid not in data_icd:
        data_read[pid]=[]
        data_icd[pid]=[]
        data_time[pid]=[]
        data_check[pid]=[]
        data_type[pid]=[]
    time=t[2].split('/')
    time='-'.join([time[2],time[1],time[0]])
    read_v2 = t[3]
    read_v3 = ''
    if len(t) >= 5:
        read_v3 = t[4]
    if read_v2 in readv2_to_ICD9 or read_v2 in readv2_to_ICD10:
        if time+'@'+read_v2 not in data_check[pid] and time not in ('1901-01-01','1902-02-02','1903-03-03','2037-07-07'):
            data_read[pid].append(read_v2)
            data_time[pid].append(time)
            data_type[pid].append('read_v2')
            data_check[pid].append(time+'@'+read_v2)         
            
    if read_v3 in readv3_to_ICD9 or read_v3 in readv3_to_ICD10:
        if time+'@'+read_v3 not in data_check[pid] and time not in ('1901-01-01','1902-02-02','1903-03-03','2037-07-07'):
            data_read[pid].append(read_v3)
            data_time[pid].append(time)
            data_type[pid].append('read_v3')
            data_check[pid].append(time+'@'+read_v3)    

max_len=0
for pid in data_icd:
    if len(data_time[pid])==0:
        continue
    tmp_time=sort_by_time(data_time[pid],data_time[pid])
    tmp_read=sort_by_time(data_read[pid],data_time[pid])
    tmp_type=sort_by_time(data_type[pid],data_time[pid])
    if len(tmp_time) > max_len:
        max_len=len(tmp_time)
    print('\t'.join([pid]+tmp_type),file=out1)
    print('\t'.join([pid]+tmp_time),file=out2)
    print('\t'.join([pid]+tmp_read),file=out3)


out1.close()
out2.close()
out3.close()

header='PID'
for xx in range(max_len):
    header=header+'\tV'+str(xx+1)

os.system("sed -i -e '1i"+header+"' "+out1f)
os.system("sed -i -e '1i"+header+"' "+out2f)
os.system("sed -i -e '1i"+header+"' "+out3f)