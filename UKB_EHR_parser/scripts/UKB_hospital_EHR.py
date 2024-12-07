import os
from tqdm import tqdm
#where are your files
import sys

main_file=sys.argv[1]
fields_annotation=sys.argv[2]

#where you want to save the ouputs
out1f='./hospital_data/temp/UKB_EHR_diagnosis_code.tsv'
out2f='./hospital_data/temp/UKB_EHR_diagnosis_codetype.tsv'
out3f='./hospital_data/temp/UKB_EHR_diagnosis_time.tsv'

out1=open(out1f,'w')
out2=open(out2f,'w')
out3=open(out3f,'w')


def sort_by_time(a,b):
    zipped = zip(a, b)
    sorted_zipped = sorted(zipped, key=lambda x: x[1].split('-'))
    a, b = zip(*sorted_zipped)
    return list(a)

field_to_ICD={}
for l in open(fields_annotation):
    l=l.rstrip().split(' ')
    field_to_ICD[l[0].split('\t')[1]]=l[-1]

code_to_time={'41202':'41262','41203':'41263','41270':'41280','41271':'41281'}


#UKB/categories/2002/ukb41808.enc_ukb/Summary_Diagnoses.tab
print('parsing hospital data')
head_to_ICD={}
head_to_time={}
idx_check=[]
max_len=0
for ix,lin in tqdm(enumerate(open(main_file))):
    lin = lin.rstrip().split('\t')
    if ix==0:
        head=lin
        for idx,x in enumerate(head):
            if x.split('.')[1] in field_to_ICD and x.split('.')[1] in code_to_time:
                idx_check.append(idx)
                head_to_ICD[x]=field_to_ICD[x.split('.')[1]]
                head_to_time[x]=head.index('f.'+code_to_time[x.split('.')[1]]+'.'+x.split('.')[2]+'.'+x.split('.')[3])
        continue
    id=[lin[0]]
    tmp1=[]
    tmp2=[]
    tmp3=[]

    check=[]
    for idx in idx_check:
        lin[idx]=lin[idx].replace('"','')
        if lin[idx] != 'NA' and lin[idx]+'@'+lin[head_to_time[head[idx]]] not in check:
            check.append(lin[idx]+'@'+lin[head_to_time[head[idx]]])
            tmp1.append(lin[idx])
            tmp2.append(head_to_ICD[head[idx]])
            tmp3.append(lin[head_to_time[head[idx]]])
    if len(tmp3)==0:
        continue
    tmp1_ = sort_by_time(tmp1,tmp3)
    tmp2_ = sort_by_time(tmp2,tmp3)
    tmp3_ = sort_by_time(tmp3,tmp3)

    if len(tmp1_) > max_len:
        max_len=len(tmp1_)
    print('\t'.join(id+tmp1_),file=out1)
    print('\t'.join(id+tmp2_),file=out2)
    print('\t'.join(id+tmp3_),file=out3)


out1.close()
out2.close()
out3.close()

header='PID'
for xx in range(max_len):
    header=header+'\tV'+str(xx+1)

os.system("sed -i -e '1i"+header+"' "+out1f)
os.system("sed -i -e '1i"+header+"' "+out2f)
os.system("sed -i -e '1i"+header+"' "+out3f)
