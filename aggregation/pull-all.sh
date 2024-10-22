

echo "Pulling demographic data..."
cd demo 
sh pull-data.sh
cd ../ 

echo "Pulling flooding data..."
cd flooding 
sh pull-data.sh
cd ../

echo "Pulling geographic data..."
cd geo
sh pull-data.sh
cd ../

echo "All data pulled successfully!"