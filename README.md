# Deep Image clustering of Mars Orbital Survey (DeIMOS)

#### Lee Ding, Joe Han, George Hu

### Getting Started
Create a folder within this called "data", and download the zip in there. Use the sample_images.py script to sample images from the zipfile. Then, run

    python3 test.py [pretrain] [kmeans]

depending on what version of the model you prefer.

### Using the VM
You can just use the browser console from the GCloud page by clicking SSH: https://console.cloud.google.com/compute/
Note: visualization will not work here as it is terminal-only. 

Guide on getting a GUI: https://medium.com/google-cloud/linux-gui-on-the-google-cloud-platform-800719ab27c5
XFCE is already installed, so only worry about the VNC server; alternatively, same some time and effort
by using Chrome Remote Desktop: https://cloud.google.com/solutions/chrome-desktop-remote-on-compute-engine

Other notes: Code files are in /[REDACTED]/DeIMOS. Use 'sudo su' to get root permissions. 
