# Street Flooding Project 
# Developer: Matt Franchi, @mattwfranchi 
# Cornell Tech 

# This script pulls the necessary geographic datasets for the creation of our analysis dataframe. 

# Create data directory
mkdir -p data

LIMIT=1000

# nyc floodnet data for september and october 2023 
# stored locally, for now, no batch export api endpoint

## NYC Flood Vulnerability Index 
## https://data.cityofnewyork.us/Environment/New-York-City-s-Flood-Vulnerability-Index/mrjc-v9pm/about_data

# need to page through this api endpoint using offset, LIMIT rows at a time 
nyc_fvi='https://data.cityofnewyork.us/resource/mrjc-v9pm.csv'
offset=0
while true; do
    wget -O data/nyc_fvi_${offset}.csv "${nyc_fvi}?\$limit=${LIMIT}&\$offset=${offset}"
    if [ $(wc -l < data/nyc_fvi_${offset}.csv) -lt $LIMIT ]; then
        break
    fi
    offset=$((offset+LIMIT))
done
# concatenate all the files together
cat data/nyc_fvi_*.csv > data/nyc_fvi.csv
# delete the intermediate files
rm data/nyc_fvi_*.csv

# 311 flooding complaints for september 29 2023 
nyc311='https://data.cityofnewyork.us/resource/erm2-nwe9.csv'
nyc311_flooding_sep29_query='?$query=SELECT%0A%20%20%60unique_key%60%2C%0A%20%20%60created_date%60%2C%0A%20%20%60closed_date%60%2C%0A%20%20%60agency%60%2C%0A%20%20%60agency_name%60%2C%0A%20%20%60complaint_type%60%2C%0A%20%20%60descriptor%60%2C%0A%20%20%60location_type%60%2C%0A%20%20%60incident_zip%60%2C%0A%20%20%60incident_address%60%2C%0A%20%20%60street_name%60%2C%0A%20%20%60cross_street_1%60%2C%0A%20%20%60cross_street_2%60%2C%0A%20%20%60intersection_street_1%60%2C%0A%20%20%60intersection_street_2%60%2C%0A%20%20%60address_type%60%2C%0A%20%20%60city%60%2C%0A%20%20%60landmark%60%2C%0A%20%20%60facility_type%60%2C%0A%20%20%60status%60%2C%0A%20%20%60due_date%60%2C%0A%20%20%60resolution_description%60%2C%0A%20%20%60resolution_action_updated_date%60%2C%0A%20%20%60community_board%60%2C%0A%20%20%60bbl%60%2C%0A%20%20%60borough%60%2C%0A%20%20%60x_coordinate_state_plane%60%2C%0A%20%20%60y_coordinate_state_plane%60%2C%0A%20%20%60open_data_channel_type%60%2C%0A%20%20%60park_facility_name%60%2C%0A%20%20%60park_borough%60%2C%0A%20%20%60vehicle_type%60%2C%0A%20%20%60taxi_company_borough%60%2C%0A%20%20%60taxi_pick_up_location%60%2C%0A%20%20%60bridge_highway_name%60%2C%0A%20%20%60bridge_highway_direction%60%2C%0A%20%20%60road_ramp%60%2C%0A%20%20%60bridge_highway_segment%60%2C%0A%20%20%60latitude%60%2C%0A%20%20%60longitude%60%2C%0A%20%20%60location%60%0AWHERE%0A%20%20(%60created_date%60%0A%20%20%20%20%20BETWEEN%20%222023-09-28T00%3A00%3A00%22%20%3A%3A%20floating_timestamp%0A%20%20%20%20%20AND%20%222023-09-30T00%3A00%3A00%22%20%3A%3A%20floating_timestamp)%0A%20%20AND%20caseless_one_of(%0A%20%20%20%20%60descriptor%60%2C%0A%20%20%20%20%22Street%20Flooding%20(SJ)%22%2C%0A%20%20%20%20%22Sewer%20Backup%20(Use%20Comments)%20(SA)%22%2C%0A%20%20%20%20%22Catch%20Basin%20Clogged%2FFlooding%20(Use%20Comments)%20(SC)%22%2C%0A%20%20%20%20%22Manhole%20Overflow%20(Use%20Comments)%20(SA1)%22%2C%0A%20%20%20%20%22Highway%20Flooding%20(SH)%22%0A%20%20)%0AORDER%20BY%20%60created_date%60%20DESC%20NULL%20FIRST'



# need to page through this api endpoint using offset, LIMIT rows at a time
wget -O data/nyc311_flooding_sep29.csv "${nyc311}${nyc311_flooding_sep29_query} LIMIT 10000 OFFSET 0"


## NYC DEP Stormwater Flooding Maps 
## https://data.cityofnewyork.us/Environment/NYC-Stormwater-Flood-Maps/9i7c-xyvv/about_data

wget -O data/nyc_stormwater_flooding_maps.zip 'https://data.cityofnewyork.us/download/9i7c-xyvv/application%2Fzip'
unzip -d data data/nyc_stormwater_flooding_maps.zip

# delete zip 
rm data/nyc_stormwater_flooding_maps.zip