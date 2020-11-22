# Postprocessing information on JLSE
The analysis directory is available at
```
/home/rmaulik/link_dir/rmaulik/DayMet/rkhs_analyses
```
Within this directory, you will see a shell_script file `jupyter_notebook.sh`, which you can qsub to grab a skylake node (check nodelist for availability). After that is done, you will need to ssh into that node from another terminal to run jupyter locally. This is possible with
```
ssh -t -t rmaulik@login.jlse.anl.gov -L 8889:localhost:8889 ssh node_name -L 8889:localhost:8889
```
Use `qstat -u rmaulik` to find what the `node_name` is (under the `Location` column). After you have `ssh`-ed into the right node, you can use `http://localhost:8889/` in your browser of choice to access the analysis directory from a jupyter session. Open a new (or previously existing notebook) as you please. Note that the JLSE `.bashrc` already has anaconda in the path and therefore you will be able to access the different kernels and virtual environments automatically (note that you need to have added the kernel to the virtual environment first).

The example shell script is
```
#!/bin/bash
#COBALT -n 1
#COBALT -A Performance
#COBALT -q skylake_8180
#COBALT -t 5:00:00

export PATH="/home/rmaulik/anaconda3/bin:$PATH"
source activate daymet_env

export https_proxy="https://proxy:3128"
export http_proxy="http://proxy:3128"
export ftp_proxy="ftp://proxy:3128"

jupyter notebook --no-browser --port=8889 --NotebookApp.token=

source deactivate
```
