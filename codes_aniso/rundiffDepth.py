#!/bin/bash
#-----------------------------------------------------------------
# Example SLURM job script to run serial applications on MOGON.
#
# This script requests one core (out of 20) on one Broadwell-node. 
# The job will have access to all the memory in the node.  Note 
# that this job will be charged as if all 20 cores were requested.
#-----------------------------------------------------------------

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --time=3:23:00
#SBATCH --mem-per-cpu=600M
#SBATCH --array=0-8
#SBATCH --output=diipaper_%A_%a.out
#SBATCH -p smp
#SBATCH -A m2_komet331hpc           
#SBATCH -c 3

declare -a ptlarrray
declare -a fractionarray
declare -a densratio
declare -a molefrac



#ptlarray=(1155108  1073100 990888 949884 908880 826872 744660 703656 621648 605328 580644  539640 531480 498636 465792)
#fractionarray=(0.84991 0.74998 0.64980 0.59984 0.54987 0.44994 0.34976 0.29979 0.19986 0.17998 0.14990 0.09993 0.08999 0.04997 0.00994)
#densratio=(1.52410 1.34490 1.16525 1.07565 0.98605 0.80685 0.62720 0.53760 0.35840 0.32274 0.26880 0.17920 0.16137 0.08960 0.01783)
#molefrac=(0.60382 0.57354 0.53816 0.51822 0.49649 0.44655 0.38545 0.34964 0.26384 0.24399 0.21185 0.15197 0.13895 0.08223 0.01752)


ptlarray=(2426844 2098608 1934592 1852584 1770372 1688364 1442136 1278120 1196112)
fractionarray=(2.39981 1.99980 1.79992 1.69998 1.59979 1.49985 1.19978 0.99990 0.89996)
densratio=(4.30305 3.58580 3.22740 3.04820 2.86855  2.68935 2.15130 1.79290 1.61370)
molefrac=(0.81143 0.78194 0.76345 0.75298 0.74151 0.72895 0.68267 0.64195 0.61740 )



#pse_sim_script="/lustre/project/m2_komet331hpc/jvishnu/Reg_network/Reg_net_bigger/time_DII.py"
pse_sim_script="/lustre/project/m2_trr146/jvishnu/Regnetbigger/Combined/Diffusiondepth.py"



module load phys/HOOMD/2.9.6-fosscuda-2019b-single
. /home/jvishnu/vhoomd/bin/activate




srun  python3 $pse_sim_script ${ptlarray[${SLURM_ARRAY_TASK_ID}]} ${fractionarray[${SLURM_ARRAY_TASK_ID}]} ${densratio[$SLURM_ARRAY_TASK_ID]}  ${molefrac[$SLURM_ARRAY_TASK_ID]} 

