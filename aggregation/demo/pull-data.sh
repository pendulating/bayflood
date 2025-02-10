mkdir -p data
# load API_KEY from key.text
# key.text should only contain your key, and be in the same directory as this script. 
API_KEY=$(cat key.txt)

for year in $(seq 2023 2023)
do

    # ACS DP05 - NYC 
    acs_dp05_md='https://api.census.gov/data/'${year}'/acs/acs5/profile/groups/DP05.json'
    wget -O data/acs${year}_dp05_md.json "${acs_dp05_md}"

    acs_dp05_template='https://api.census.gov/data/'${year}'/acs/acs5/profile?get=group(DP05)&for=tract:*&in=state:36&in=county:005,047,061,081,085'
    wget -O data/acs${year}_dp05.json "${acs_dp05_template}&key=${API_KEY}"

    # Internet Access 
    acs_s2801_md='https://api.census.gov/data/'${year}'/acs/acs5/subject/groups/S2801.json'
    wget -O data/acs${year}_s2801_md.json "${acs_s2801_md}"

    acs_s2801_template='https://api.census.gov/data/'${year}'/acs/acs5/subject?get=group(S2801)&for=tract:*&in=state:36&in=county:005,047,061,081,085'
    wget -O data/acs${year}_s2801.json "${acs_s2801_template}&key=${API_KEY}"

    # Median Household Income
    acs_s1901_md='https://api.census.gov/data/'${year}'/acs/acs5/subject/groups/S1901.json'
    wget -O data/acs${year}_s1901_md.json "${acs_s1901_md}"

    acs_s1901_template='https://api.census.gov/data/'${year}'/acs/acs5/subject?get=group(S1901)&for=tract:*&in=state:36&in=county:005,047,061,081,085'
    wget -O data/acs${year}_s1901.json "${acs_s1901_template}&key=${API_KEY}"

    # Educational Attainment 
    acs_s1501_md='https://api.census.gov/data/'${year}'/acs/acs5/subject/groups/S1501.json'
    wget -O data/acs${year}_s1501_md.json "${acs_s1501_md}"

    acs_s1501_template='https://api.census.gov/data/'${year}'/acs/acs5/subject?get=group(S1501)&for=tract:*&in=state:36&in=county:005,047,061,081,085'
    wget -O data/acs${year}_s1501.json "${acs_s1501_template}&key=${API_KEY}"

    # Limited English Speaking Households 
    acs_s1602_md='https://api.census.gov/data/'${year}'/acs/acs5/subject/groups/S1602.json'
    wget -O data/acs${year}_s1602_md.json "${acs_s1602_md}"

    acs_s1602_template='https://api.census.gov/data/'${year}'/acs/acs5/subject?get=group(S1602)&for=tract:*&in=state:36&in=county:005,047,061,081,085'
    wget -O data/acs${year}_s1602.json "${acs_s1602_template}&key=${API_KEY}"

   

done