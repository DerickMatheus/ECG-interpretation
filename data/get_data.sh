wget "https://zenodo.org/record/3625027/files/data.zip?download=1"
wget "https://zenodo.org/record/3625018/files/model.zip?download=1"
unzip 'data.zip?download=1'
unzip 'model.zip?download=1'
mv model/* ../data
mv data/* ../data
rm -r model
rm -r data
rm 'data.zip?download=1'
rm 'model.zip?download=1'

