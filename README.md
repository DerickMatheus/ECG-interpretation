# ECG-interpretation
<h2> How To Run </h2>

<h4> Downloading Data </h4>
On the "data" folder, you'll find a file named "get_data.sh". Run <i>sh get_data.sh</i> on your terminal and all
the data will be downloaded and extracted.

<h4> Matching the Requirements </h4>
On the main folder, there's a file called "requirements.yml". Create a Conda Environment from that file and you
should be ready to go. <br/>
Further instructions can be found in the link below: <br/>
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html.

<h4> Running the Code </h4>

In the main folder, go for <i>python run.py</i> on your terminal and you'll run the code with default values.
The default values are as follows:

<br/><i>
val_traces = 'data/ecg_tracings.hdf5' <br/>
model  = 'data/model.hdf5' <br/>
real   = None <br/>
noise  = None <br/>
sim    = 100 <br/>
id_ecg = "all" <br/>
</i>

To run with values other than the default, you can type:<br/>
<i>python run.py with 'a=value1' 'b="value2"' 'c=value3'</i>

For example, to run for id_ecg = 1, number of simulations = 50 and another dataset, you would do: <br/>
<i>python run.py with 'id_ecg=1' 'sim=100' 'val_traces="path/to/dataset.hdf5"'</i> <br/>
<br/>
Further instructions can be found in the link below:<br/>
https://sacred.readthedocs.io/en/stable/command_line.html

<h2> Code Output </h2>

<h4> Graphs </h4>

A graphs folder will be created in the main directory, containing the following graphs:

<i>
real.png <br/>
noiseAV_rate.png <br/> 
noiseqrs.png <br/>
noisest.png <br/>
noiseaxis.png  <br/>
noiseqt.png  <br/>
noiserandom30.png  <br/>
noiset.png <br/>
noisep.png  <br/>
noiser.png  <br/>
noiserandom50.png <br/>
noisepr.png  <br/>
noiserandom.png  <br/>
noiserhythm.png <br/>
noiseq.png   <br/>
noises.png <br/>
</i>
<br/>
The <i>real.png</i> file represents the plot for the original ECG and all the other ones represents how the ECG looks
with each noise applied.

<h4> Results </h4>

A output_result folder will be created in the main directory, containing two other folders:

<h5>tests</h5>

The tests folder contains one file for each type of noise that was applied. <br/>
The files contain one vector for each simulation that was made and these vectors represent how much impact the noise
caused in the judgement of the classification model.

<h5>mean</h5>

The mean folder contains one file for each execution of the code, identified by the eletrocardiogram id. <br/>
The files are in R format and contains one vector for each type of noise that was applied.<br/>
These vectors represent the mean of how much impact the noise caused in the judgement of the classification model for every simulation. That is, the vectors in this file are the mean of the vectors previously mentioned on the tests folder.
