import os
import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import sys
basic_file=sys.argv[1]

#where you want to save the ouputs
out1f='./mergered_data/UKB_EHR_diagnosis_code.tsv'
out2f='./mergered_data/UKB_EHR_diagnosis_codetype.tsv'
out3f='./mergered_data/UKB_EHR_diagnosis_time.tsv'
out4f='./mergered_data/UKB_EHR_diagnosis_age.tsv'

out1=open(out1f,'w')
out2=open(out2f,'w')
out3=open(out3f,'w')
out4=open(out4f,'w')

birth={}
for l in open(basic_file):
    l=l.rstrip()
    t=l.split('\t')
    if t[0]=='f.eid' or t[2]=='NA':
        continue
    birth[t[0]]=datetime.datetime.strptime(t[2]+'-'+t[3],"%Y-%m")

def sort_by_time(a,b):
    zipped = zip(a, b)
    sorted_zipped = sorted(zipped, key=lambda x: x[1].split('-'))
    a, b = zip(*sorted_zipped)
    return list(a)


hospital_code={}
hospital_type={}
hospital_time={}
hospital_age={}
interleaved_time={}
for l in open('./hospital_data/temp/UKB_EHR_diagnosis_code.tsv'):
    l=l.rstrip()
    t=l.split('\t')
    if t[0]=='PID':
        continue
    hospital_code[t[0]]=t[1:]

for l in open('./hospital_data/temp/UKB_EHR_diagnosis_codetype.tsv'):
    l=l.rstrip()
    t=l.split('\t')
    if t[0]=='PID':
        continue
    hospital_type[t[0]]=t[1:]

for l in open('./hospital_data/temp/UKB_EHR_diagnosis_time.tsv'):
    l=l.rstrip()
    t=l.split('\t')
    if t[0]=='PID':
        continue
    hospital_time[t[0]]=t[1:]
    interleaved_time[t[0]]=[x+'-02' for x in t[1:]]

for l in open('./primary_data/temp/UKB_EHR_diagnosis_readcode.tsv'):
    l=l.rstrip()
    t=l.split('\t')
    if t[0]=='PID':
        continue
    if t[0] in hospital_code:
        hospital_code[t[0]].extend(t[1:])
    else:
        hospital_code[t[0]]=t[1:]

for l in open('./primary_data/temp/UKB_EHR_diagnosis_codetype.tsv'):
    l=l.rstrip()
    t=l.split('\t')
    if t[0]=='PID':
        continue
    if t[0] in hospital_type:
        hospital_type[t[0]].extend(t[1:])
    else:
        hospital_type[t[0]]=t[1:]

for l in open('./primary_data/temp/UKB_EHR_diagnosis_time.tsv'):
    l=l.rstrip()
    t=l.split('\t')
    if t[0]=='PID':
        continue
    if t[0] in hospital_time:
        hospital_time[t[0]].extend(t[1:])
        interleaved_time[t[0]].extend([x+'-01' for x in t[1:]])
    else:
        hospital_time[t[0]]=t[1:]
        interleaved_time[t[0]]=[x+'-01' for x in t[1:]]

max_len=0
max_age=0
print('merge hospital and primary data')
for pid in tqdm(hospital_code):
    tmp_code=sort_by_time(hospital_code[pid],interleaved_time[pid])
    tmp_type=sort_by_time(hospital_type[pid],interleaved_time[pid])
    tmp_time=sort_by_time(hospital_time[pid],interleaved_time[pid])
    diff=[relativedelta(datetime.datetime.strptime(x,"%Y-%m-%d"),birth[pid]) for x in tmp_time]
    tmp_age=[str(x.years*12+x.months) for x in diff]
    max_age=max([max_age]+[x.years*12+x.months for x in diff])

    if len(tmp_time) > max_len:
        max_len=len(tmp_time)
    print('\t'.join([pid]+tmp_code),file=out1)
    print('\t'.join([pid]+tmp_type),file=out2)
    print('\t'.join([pid]+tmp_time),file=out3)
    print('\t'.join([pid]+tmp_age),file=out4)


out1.close()
out2.close()
out3.close()
out4.close()

header='PID'
for xx in range(max_len):
    header=header+'\tV'+str(xx+1)

os.system("sed -i -e '1i"+header+"' "+out1f)
os.system("sed -i -e '1i"+header+"' "+out2f)
os.system("sed -i -e '1i"+header+"' "+out3f)
os.system("sed -i -e '1i"+header+"' "+out4f)