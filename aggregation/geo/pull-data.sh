# Street Flooding Project 
# Developer: Matt Franchi, @mattwfranchi 
# Cornell Tech 

# This script pulls the necessary geographic datasets for the creation of our analysis dataframe. 

# Create data directory
mkdir -p data

# Function to fetch all features from ArcGIS Feature Server
fetch_arcgis_features() {
    local url="$1"
    local output_file="$2"
    local where_clause="${3:-1=1}" # Default to 1=1 if not provided
    local offset=0
    local limit=2000  # ArcGIS standard limit is often 2000, sometimes 1000
    local temp_file="temp_features.geojson"
    
    echo "Fetching data for $output_file..."

    # Encode where_clause
    encoded_where=$(python3 -c "import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1]))" "$where_clause")
    
    # Initialize output file
    echo '{ "type": "FeatureCollection", "features": [' > "$output_file"
    
    first_batch=true
    
    while true; do
        echo "Fetching records with offset $offset..."
        
        # Construct query URL
        # Use resultOffset and resultRecordCount for pagination
        # outSR=4326 ensures WGS84 coordinates
        # f=geojson returns GeoJSON format
        
        query_url="${url}/query?where=${encoded_where}&outFields=*&outSR=4326&f=geojson&resultOffset=${offset}&resultRecordCount=${limit}"
        
        # Fetch batch - use quotes around URL to prevent shell expansion
        wget -q -O "$temp_file" "$query_url"
        
        # Check if we got features
        feature_count=$(grep -o '"type":"Feature"' "$temp_file" | wc -l)
        
        if [ "$feature_count" -eq 0 ]; then
            echo "No more features found or error in query."
            # Check if file contains error
            if grep -q "error" "$temp_file"; then
                echo "Error response from server:"
                cat "$temp_file"
            fi
            break
        fi
        
        echo "Retrieved $feature_count features."
        
        # Extract features array content
        python3 -c "import json, sys; data = json.load(open('$temp_file')); print(json.dumps(data['features'])[1:-1])" > batch_features.json
        
        if [ "$first_batch" = true ]; then
            cat batch_features.json >> "$output_file"
            first_batch=false
        else
            echo "," >> "$output_file"
            cat batch_features.json >> "$output_file"
        fi
        
        if [ "$feature_count" -lt "$limit" ]; then
            echo "Finished fetching all records."
            break
        fi
        
        offset=$((offset + limit))
    done
    
    # Close the JSON object
    echo ']}' >> "$output_file"
    
    # Clean up
    rm "$temp_file" batch_features.json
    echo "Saved to $output_file"
}

# Geographic Boundaries 

## 2020 NYC Census Tracts, water areas clipped
fetch_arcgis_features 'https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/NYC_Census_Tracts_for_2020_US_Census/FeatureServer/0' 'data/ct-nyc-2020.geojson'

## 2020 NYC Census Tracts, including water areas
fetch_arcgis_features 'https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/NYC_Census_Tracts_for_2020_US_Census_Water_Included/FeatureServer/0' 'data/ct-nyc-wi-2020.geojson'

## 2020 NYC Census Blocks, including water areas 
fetch_arcgis_features 'https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/NYC_Census_Blocks_for_2020_US_Census_Water_Included/FeatureServer/0' 'data/cb-nyc-wi-2020.geojson'

## 2020 NYC Census Blocks, water areas clipped
fetch_arcgis_features 'https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/NYC_Census_Blocks_for_2020_US_Census/FeatureServer/0' 'data/cb-nyc-2020.geojson'

## 2020 NYC Census Block Groups (from Census TIGERweb)
# Layer 11 is Census Block Groups (2020) in Census2020/Tracts_Blocks
# Filter for NYC Counties: Bronx(005), Kings(047), New York(061), Queens(081), Richmond(085)
# State: 36 (New York)
fetch_arcgis_features 'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/11' 'data/cbg-nyc-2020.geojson' "STATE='36' AND COUNTY IN ('005','047','061','081','085')"

## NYC Integer 1 foot Digital Elevation Model Raster 
wget -O data/nyc-1ft-dem.zip 'https://sa-static-customer-assets-us-east-1-fedramp-prod.s3.amazonaws.com/data.cityofnewyork.us/NYC_DEM_1ft_Int.zip'
unzip -d data data/nyc-1ft-dem.zip

# delete zip
rm data/nyc-1ft-dem.zip
