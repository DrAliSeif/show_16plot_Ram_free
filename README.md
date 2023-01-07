# show_16plot_Ram_free
show plot ram free 16 plot for correlation, sync, phase circle, average and total phase and frequency and color phase node

#  Python

# Imports
_________________
### usage Ram
```ruby
import os                                    #for create empty folder and calculate ram
import psutil                                #calculate ram
process = psutil.Process(os.getpid())
print('Ram usage: {:.2f} MB'.format(process.memory_info().rss/1000000))
```
### other library

```ruby
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib as mpl
import math
import cmath
import pandas as pd
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import matplotlib.colors
import matplotlib.patches as mpatches 
from matplotlib.collections import PatchCollection
import time
import gc
print('Ram usage: {:.2f} MB'.format(process.memory_info().rss/1000000))

```
# not change in any case
___________________
```ruby
start_time=time.time()
Number_of_node=1000
Number_of_step=2001#becuase start since 0
font1 = {'family': 'serif', 'color': 'blue', 'size': 190}
font2 = {'family': 'serif', 'color': 'darkred', 'size': 12}
Color_map_hsv='hsv'
Color_map_brg='brg'
def draw_self_loop(center, radius, facecolor='#2693de', edgecolor='#000000', theta1=0, theta2=180):
    
    # Add the ring
    rwidth = 0.1
    ring = mpatches.Wedge(center, radius, theta1, theta2, width=rwidth)

    # Triangle edges
    offset = 0.0
    xcent  = center[0] - radius + (rwidth/2)
    left   = [xcent - offset, center[1]]
    right  = [xcent + offset, center[1]]
    bottom = [(left[0]+right[0])/2., center[1]-0.05]
    arrow  = plt.Polygon([left, right, bottom, left])

    p = PatchCollection(
        [ring, arrow], 
        edgecolor = edgecolor, 
        facecolor = facecolor
    )
    ax7.add_collection(p)
print("Time: {:.3f} sec".format(time.time()-start_time))
print('Ram usage: {:.2f} MB'.format(process.memory_info().rss/1000000))

```
# not change exept change variable & adrress
# also define all arrey
______________
```ruby
start_time=time.time()
address=r'G:/New Two jump first order transition (Not Scale)/Cpp/Change a algorithm seif/'
address_backward=r'G:/New Two jump first order transition (Not Scale)/Cpp/Change a algorithm seif Backward/'

dW=0.8
dw_left_cut=277
dw_Right_cut=723

landa=10
timeforloop_Average_sync = [0 for y in range(301)] #timeforloop total sync
Layer2_Average_sync = [0 for y in range(301)] #sync total for change coupling
Layer2_Average_sync_backward = [0 for y in range(301)] #sync total for change coupling
Matrix_sor_0_2PI = [[0 for x in range(Number_of_node)] for y in range(Number_of_step)] 
Corolation = [[0 for x in range(Number_of_node)] for y in range(Number_of_node)] 
################################################################################
#                           Read data omega                            # START #
################################################################################
number_omega=[0 for y in range(Number_of_node)]#for layer 1 omega
omega=[0 for y in range(Number_of_node)]#for layer 1 omega
arr_xomega_left=[0 for y in range(dw_left_cut)]#for layer 2 left part omega
arr_yomega_left=[0 for y in range(dw_left_cut)]#for layer 2 left part omega
arr_xomega_mid=[0 for y in range(dw_Right_cut-dw_left_cut)]#for layer 2 mid part omega
arr_yomega_mid=[0 for y in range(dw_Right_cut-dw_left_cut)]#for layer 2 mid part omega
arr_xomega_right=[0 for y in range(Number_of_node-dw_Right_cut)]#for layer 2 right part omega
arr_yomega_right=[0 for y in range(Number_of_node-dw_Right_cut)]#for layer 2 right part omega
data_omega=np.loadtxt(address+'Example/W=Natural frequency/'+str(dW)+'Layer2.txt')
#for layer 1
for x in range(0,Number_of_node):
    number_omega[x]=x+1
    omega[x]=((x)/999)-0.5  
#for layer 2 left part
for x in range(0,dw_left_cut):
    arr_xomega_left[x]=x+1
    arr_yomega_left[x]=data_omega[x]   
#for layer 2 mid part
for x in range(dw_left_cut,dw_Right_cut):
    arr_xomega_mid[x-dw_left_cut]=x+1
    arr_yomega_mid[x-dw_left_cut]=data_omega[x]  
#for layer 2 right part
for x in range(dw_Right_cut,Number_of_node):
    arr_xomega_right[x-dw_Right_cut]=x+1
    arr_yomega_right[x-dw_Right_cut]=data_omega[x]
################################################################################
#                           Read data omega                            #  END  #
################################################################################
G = nx.cycle_graph(Number_of_node+1)
Scale=[0 for y in range(Number_of_node+1)]
Scale[0]=0
for x in range(1, Number_of_node):
    Scale[x]=1
Scale[Number_of_node]=0
pos = nx.circular_layout(G,scale=Scale)
#Part syncroney
Number_total=[0 for y in range(Number_of_step)]#total sync
Total_sync=[0 for y in range(Number_of_step)]#total sync
left_sync=[0 for y in range(Number_of_step)]
mid_sync=[0 for y in range(Number_of_step)]
right_sync=[0 for y in range(Number_of_step)]
################################################################################
#After timeforloop steps
Color_of_node=[0 for y in range(Number_of_node+1)]
#PLOTS
#Left Phase circle                                                                  1
arrays_node_rad_left=[0 for y in range(dw_left_cut+2)]
arrays_node_rad0_left=[0 for y in range(dw_left_cut+2)]
arrays_node_radcolor_left=[0 for y in range(dw_left_cut+2)]
#mid Phase circle                                                                   2
arrays_node_rad_mid=[0 for y in range(dw_Right_cut-dw_left_cut+2)]
arrays_node_rad0_mid=[0 for y in range(dw_Right_cut-dw_left_cut+2)]
arrays_node_radcolor_mid=[0 for y in range(dw_Right_cut-dw_left_cut+2)]
#Right Phase circle                                                                 3
arrays_node_rad_Righ=[0 for y in range(Number_of_node-dw_Right_cut+2)]
arrays_node_rad0_Righ=[0 for y in range(Number_of_node-dw_Right_cut+2)]
arrays_node_radcolor_Righ=[0 for y in range(Number_of_node-dw_Right_cut+2)]
#Distribution function circle                                                       4
number_of_circ=len(arrays_node_rad_mid)-1
condition=[]
for timeforloop in range(0, 378,18):
    condition.append(timeforloop*(math.pi/180)) 
values = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]#[0,1] radiation
number_of_circ_left=len(arrays_node_rad_left)-1
values_left = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]#[0,1] radiation
number_of_circ_Righ=len(arrays_node_rad_Righ)-1
values_Righ = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]#[0,1] radiation
angel=[]
for timeforloop in range(0, 360,18):
    angel.append(timeforloop)    

theta = [0.155,0.47,0.785,1.1,1.415 ,1.726 ,2.045 ,2.36 ,2.67 ,2.986,3.298 ,3.617 ,3.93,4.243 ,4.558 ,4.875,5.185,5.5,5.815,6.125]
width = [0.31,0.31,0.31,0.31,0.31 ,0.31 ,0.31,0.31,0.31,0.31,0.31,0.31,0.31,0.31,0.31,0.31,0.31,0.31,0.31,0.31]

data_mod_2pi_left_sorted=[0 for y in range(number_of_circ_left-1)]
data_mod_2pi_mid_sorted=[0 for y in range(number_of_circ-1)]
data_mod_2pi_right_sorted=[0 for y in range(number_of_circ_Righ-1)]


average_colors=[1 ,0 ,-1]#[left , mid, right]
average_colors += average_colors[:1]#first connect to last


average_angels=[0.4,1.5,4.6,0.4]
average_values=[1,0.4,0.7,1]

data_mod_2pi=[0 for y in range(Number_of_node)]

#subplot9
Frequency = [[0 for x in range(Number_of_step-2)] for y in range(Number_of_node)] #[soton][satr] // #[node][timeforloop]  //[0,999][0,2000]
timeforloop_Fre=[0 for y in range(Number_of_step-2)]

        
#average
f_total=[0 for y in range(Number_of_step-2)]

f_left=[0 for y in range(Number_of_step-2)]

f_mid=[0 for y in range(Number_of_step-2)]

f_right=[0 for y in range(Number_of_step-2)]
print("Time: {:.3f} sec".format(time.time()-start_time))
print('Ram usage: {:.2f} MB'.format(process.memory_info().rss/1000000))

```
# transformative
___________
### α = Degree frustration
```ruby
start_time=time.time()
#################################################################################################
##########                                                                             ##########
##########                                 DEGREE                                      ##########
##########                                                                             ##########
#################################################################################################
Degree=1.57 #degree alpha
Degree_data='Degree_Radian='+str(Degree)#string degree
################################################################################
#                           Read data Average_Syncrony                 # START #
################################################################################
data_Average_sync=np.loadtxt(address+'Save/Average_Syncrony(couplig_SyncL1_SyncL2)/'+Degree_data+'.txt')#read total sync
data_Average_sync_backward=np.loadtxt(address_backward+'Save/Average_Syncrony(couplig_SyncL1_SyncL2)/'+Degree_data+'.txt')#read total sync

for x in range(0, 301):#Number_of_node+1-2    //301
    timeforloop_Average_sync[x]=data_Average_sync[x][0]
    Layer2_Average_sync[x]=data_Average_sync[x][2]
    Layer2_Average_sync_backward[300-x]=data_Average_sync_backward[x][2]


################################################################################
#                           Read data Average_Syncrony                 #  END  #
################################################################################
print("Time: {:.3f} sec".format(time.time()-start_time))
print('Ram usage: {:.2f} MB'.format(process.memory_info().rss/1000000))

```
### σ = Coupling
```ruby
start_time=time.time()
#################################################################################################
##########                                                                             ##########
##########                                COUPLING                                     ##########
##########                                                                             ##########
#################################################################################################
##print("after coupling time: {:.3f} sec".format(time.time()-start_time))
##start_time=time.time()

copling=0 #coupling in layers
file_name=Degree_data+'_copling='+str(copling)+'layer2(time)VS(Node=)'
path=address+"Plot/"+file_name+"/"

################################################################################
#                           Read data Phase nodes                      # START #
################################################################################
data=np.loadtxt(address+'Save/Phases/'+file_name+'.txt')
#read calculation Matrix_sor_0_2PI
for timeforloop in range(0, Number_of_step):
    for y in range(1, Number_of_node+1):#for timeforloop step  
        Matrix_sor_0_2PI[timeforloop][y-1]=data[timeforloop][y]%(2*math.pi)
################################################################################
#                           Read data Phase nodes                      #  END  #
################################################################################
################################################################################
#                                   Part syncroney                     # START #
################################################################################
for timeforloop in range(0, Number_of_step):
    rc = 0.0
    rs = 0.0
    for y in range(0, Number_of_node):#for timeforloop step  
        rc=rc+math.cos(Matrix_sor_0_2PI[timeforloop][y])
        rs=rs+math.sin(Matrix_sor_0_2PI[timeforloop][y])
    Number_total[timeforloop]=(timeforloop/100)#doroste
    Total_sync[timeforloop]=(math.sqrt(math.pow(rc, 2) + math.pow(rs, 2)) / (Number_of_node))

for timeforloop in range(0, Number_of_step):
    rc = 0.0
    rs = 0.0
    for y in range(0,dw_left_cut):#for timeforloop step  
        rc=rc+math.cos(Matrix_sor_0_2PI[timeforloop][y])
        rs=rs+math.sin(Matrix_sor_0_2PI[timeforloop][y])
    left_sync[timeforloop]=(math.sqrt(math.pow(rc, 2) + math.pow(rs, 2)) / (dw_left_cut))    


for timeforloop in range(0, Number_of_step):
    rc = 0.0
    rs = 0.0
    for y in range(dw_left_cut,dw_Right_cut):#for timeforloop step  
        rc=rc+math.cos(Matrix_sor_0_2PI[timeforloop][y])
        rs=rs+math.sin(Matrix_sor_0_2PI[timeforloop][y])
    mid_sync[timeforloop]=(math.sqrt(math.pow(rc, 2) + math.pow(rs, 2)) / (dw_Right_cut-dw_left_cut))    


for timeforloop in range(0, Number_of_step):
    rc = 0.0
    rs = 0.0
    for y in range(dw_Right_cut,Number_of_node):#for timeforloop step  
        rc=rc+math.cos(Matrix_sor_0_2PI[timeforloop][y])
        rs=rs+math.sin(Matrix_sor_0_2PI[timeforloop][y])
    right_sync[timeforloop]=(math.sqrt(math.pow(rc, 2) + math.pow(rs, 2)) / (Number_of_node-dw_Right_cut))    
################################################################################
#                                   Part syncroney                     #  END  #
################################################################################
for timeforloop in range(1,2000):                                                              #data[timeforloop][node]
    timeforloop_Fre[timeforloop-1]=timeforloop/100                                                            #Frequency[node][timeforloop]
    for num in range(0,1000):
        Frequency[num][timeforloop-1]=round((data[timeforloop+1][num+1]-data[timeforloop-1][num+1])*50,2)
        #print(str(Frequency[num][timeforloop-1])+'\t'+str(data[timeforloop+1][num+1])+'\t'+str(data[timeforloop-1][num+1]))



for timeforloop in range(0,1999):                                                              #data[timeforloop][node]
    cont_total=0
    cont_left=0
    cont_mid=0
    cont_right=0
    for num in range(0,Number_of_node):#for timeforloop in range(0,dw_left_cut):
        cont_total=cont_total+1
        f_total[timeforloop]=f_total[timeforloop]+Frequency[num][timeforloop]
    for num in range(0,dw_left_cut):#for timeforloop in range(0,dw_left_cut):
        cont_left=cont_left+1
        f_left[timeforloop]=f_left[timeforloop]+Frequency[num][timeforloop]
    for num in range(dw_left_cut,dw_Right_cut):#for timeforloop in range(dw_left_cut,dw_Right_cut):
        cont_mid=cont_mid+1
        f_mid[timeforloop]=f_mid[timeforloop]+Frequency[num][timeforloop]
    for num in range(dw_Right_cut,Number_of_node):#for timeforloop in range dw_Right_cut,Number_of_node:
        cont_right=cont_right+1
        f_right[timeforloop]=f_right[timeforloop]+Frequency[num][timeforloop]
for timeforloop in range(0,1999):
    f_total[timeforloop]=f_total[timeforloop]/cont_total
    f_left[timeforloop]=f_left[timeforloop]/cont_left
    f_mid[timeforloop]=f_mid[timeforloop]/cont_mid
    f_right[timeforloop]=f_right[timeforloop]/cont_right

print("Time: {:.3f} sec".format(time.time()-start_time))
print('Ram usage: {:.2f} MB'.format(process.memory_info().rss/1000000))

```
### t = time step
```ruby
#################################################################################################
##########                                                                             ##########
##########                              timeforloop STEPS                              ##########
##########                                                                             ##########
#################################################################################################
start_time=time.time()
step=0
#calculation Color_of_node (sort in 0 to 2PI)
Color_of_node[0]=0
for x in range(1, Number_of_node):#Number_of_node+1-2
    Color_of_node[x]=data[step][x]%(2*math.pi)
Color_of_node[Number_of_node]=2*math.pi
#read calculation Corolation
for x in range(0, Number_of_node):
    for y in range(0, Number_of_node):
        Corolation[x][y]=math.cos(Matrix_sor_0_2PI[step][x]-Matrix_sor_0_2PI[step][y])#[satr][soton]
print("Time: {:.3f} sec".format(time.time()-start_time))
print('Ram usage: {:.2f} MB'.format(process.memory_info().rss/1000000))


```
# PLOTS
```ruby
#****************************************************************************************************                +--------+
#****************************************************************      Left cyrcle phase     ********                + Define +
#****************************************************************************************************                +--------+
def plot_phase_left(ax1):
    arrays_node_rad_left[0]=0#teta
    arrays_node_rad0_left[0]=0#radiation
    arrays_node_radcolor_left[0]=-1#color
    for j in range(0,dw_left_cut+1):
        arrays_node_rad_left[j+1]=Matrix_sor_0_2PI[step][j]
        arrays_node_rad0_left[j+1]=1
        arrays_node_radcolor_left[j+1]=1
    arrays_node_rad_left[dw_left_cut+1]=0
    arrays_node_rad0_left[dw_left_cut+1]=0
    arrays_node_radcolor_left[dw_left_cut+1]=1
    ax1.scatter(arrays_node_rad_left, arrays_node_rad0_left, c=arrays_node_radcolor_left, cmap='brg')
    ax1.scatter(arrays_node_rad_left[200], arrays_node_rad0_left[200], color="#cc0066",s=100,edgecolors='k')
    ax1.set_rgrids([1.1])
    ax1.axes.get_yaxis().set_ticks([])#****************************************************************************************************                +--------+
#****************************************************************      Mid cyrcle phase      ********                + Define +
#****************************************************************************************************                +--------+
def plot_phase_mid(ax2):#, fontsize=12
    arrays_node_rad_mid[0]=0#teta
    arrays_node_rad0_mid[0]=0#radiation
    arrays_node_radcolor_mid[0]=-1#color
    for j in range(dw_left_cut,dw_Right_cut+1):
        arrays_node_rad_mid[j+1-dw_left_cut]=Matrix_sor_0_2PI[step][j]
        arrays_node_rad0_mid[j+1-dw_left_cut]=1
        arrays_node_radcolor_mid[j+1-dw_left_cut]=0
    arrays_node_rad_mid[dw_Right_cut-dw_left_cut+1]=0
    arrays_node_rad0_mid[dw_Right_cut-dw_left_cut+1]=0
    arrays_node_radcolor_mid[dw_Right_cut-dw_left_cut+1]=1
    ax2.scatter(arrays_node_rad_mid, arrays_node_rad0_mid, c=arrays_node_radcolor_mid, cmap='brg')
    ax2.scatter(arrays_node_rad_mid[500-dw_left_cut], arrays_node_rad0_mid[500-dw_left_cut], color="#333333",s=100,edgecolors='k')
    ax2.set_rgrids([1.1], labels=None)#ax2.grid(which="major")#
    ax2.axes.get_yaxis().set_ticks([])#****************************************************************************************************                +--------+
#****************************************************************      Right cyrcle phase    ********                + Define +
#****************************************************************************************************                +--------+
def plot_phase_Right(ax3):
    arrays_node_rad_Righ[0]=0#teta
    arrays_node_rad0_Righ[0]=0#radiation
    arrays_node_radcolor_Righ[0]=-1#color
    for j in range(dw_Right_cut,Number_of_node):
        arrays_node_rad_Righ[j+1-dw_Right_cut]=Matrix_sor_0_2PI[step][j]
        arrays_node_rad0_Righ[j+1-dw_Right_cut]=1
        arrays_node_radcolor_Righ[j+1-dw_Right_cut]=-1
    arrays_node_rad_Righ[Number_of_node-dw_Right_cut+1]=0
    arrays_node_rad0_Righ[Number_of_node-dw_Right_cut+1]=0
    arrays_node_radcolor_Righ[Number_of_node-dw_Right_cut+1]=1
    ax3.scatter(arrays_node_rad_Righ, arrays_node_rad0_Righ, c=arrays_node_radcolor_Righ, cmap='brg')
    ax3.scatter(arrays_node_rad_Righ[800-dw_Right_cut], arrays_node_rad0_Righ[800-dw_Right_cut], color="#ff6600",s=100,edgecolors='k')
    ax3.set_rgrids([1.1])#ax3.grid(which="major")#
    ax3.axes.get_yaxis().set_ticks([])
#****************************************************************************************************                +--------+
#***************      Distribution function circle phase and average total sync and phase    ********                + Define +
#****************************************************************************************************                +--------+
def plot_Distribution(ax4):
    for timeforloop in range(1,number_of_circ):
        if 0>arrays_node_rad_mid[timeforloop]:
            arrays_node_rad_mid[timeforloop]=arrays_node_rad_mid[timeforloop]+2*math.pi
        for cone in range(0,20):
            if condition[cone]<=arrays_node_rad_mid[timeforloop]:
                if condition[cone+1]>=arrays_node_rad_mid[timeforloop]:
                    values[cone]=values[cone]+1
    for cone in range(0,20):
        values[cone]=values[cone]/(number_of_circ-1)
    #################################################################################
    for timeforloop in range(1,number_of_circ_left):
        if 0>arrays_node_rad_left[timeforloop]:
            arrays_node_rad_left[timeforloop]=arrays_node_rad_left[timeforloop]+2*math.pi
        for cone in range(0,20):
            if condition[cone]<=arrays_node_rad_left[timeforloop]:
                if condition[cone+1]>=arrays_node_rad_left[timeforloop]:
                    values_left[cone]=values_left[cone]+1
    for cone in range(0,20):
        values_left[cone]=values_left[cone]/(number_of_circ_left-1)    
    #################################################################################
    for timeforloop in range(1,number_of_circ_Righ):
        if 0>arrays_node_rad_Righ[timeforloop]:
            arrays_node_rad_Righ[timeforloop]=arrays_node_rad_Righ[timeforloop]+2*math.pi
        for cone in range(0,20):
            if condition[cone]<=arrays_node_rad_Righ[timeforloop]:
                if condition[cone+1]>=arrays_node_rad_Righ[timeforloop]:
                    values_Righ[cone]=values_Righ[cone]+1
    for cone in range(0,20):
        values_Righ[cone]=values_Righ[cone]/(number_of_circ_Righ-1)    
    #####################################################################################
    #############################      Average     ######################################
    ##@@@@@@@@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ last average phase
    for node in range(0,Number_of_node):
        data_mod_2pi[node]=data[step][node+1]%(2*math.pi)
    for i in range (0,len(data_mod_2pi_left_sorted)):
        data_mod_2pi_left_sorted[i]=data_mod_2pi[i]
    for i in range (dw_left_cut,dw_Right_cut):
        data_mod_2pi_mid_sorted[i-dw_left_cut]=data_mod_2pi[i]
    for i in range (dw_Right_cut,Number_of_node):
        data_mod_2pi_right_sorted[i-dw_Right_cut]=data_mod_2pi[i]
    #left
    resolt_rad=0
    resolt_degree=0
    for i in range(0,len(data_mod_2pi_left_sorted)):
        resolt_rad=resolt_rad+cmath.exp(complex(0, data_mod_2pi_left_sorted[i]))
    resolt_rad=resolt_rad/len(data_mod_2pi_left_sorted)
    if resolt_rad.real<0:
        resolt_rad=cmath.atan(resolt_rad.imag/resolt_rad.real)#resolt_degree=resolt_rad*180/math.pi
        average_angels[0]=resolt_rad.real+math.pi
    else:
        resolt_rad=cmath.atan(resolt_rad.imag/resolt_rad.real)#resolt_degree=resolt_rad*180/math.pi
        average_angels[0]=resolt_rad.real
    #mid
    resolt_rad=0
    resolt_degree=0
    for i in range(0,len(data_mod_2pi_mid_sorted)):
        resolt_rad=resolt_rad+cmath.exp(complex(0, data_mod_2pi_mid_sorted[i]))
    resolt_rad=resolt_rad/len(data_mod_2pi_mid_sorted)
    #print(resolt_rad)
    if resolt_rad.real<0:
        resolt_rad=cmath.atan(resolt_rad.imag/resolt_rad.real)#resolt_degree=resolt_rad*180/math.pi
        average_angels[1]=resolt_rad.real+math.pi
    else:
        resolt_rad=cmath.atan(resolt_rad.imag/resolt_rad.real)#resolt_degree=resolt_rad*180/math.pi
        average_angels[1]=resolt_rad.real
    #Right
    resolt_rad=0
    resolt_degree=0
    for i in range(0,len(data_mod_2pi_right_sorted)):
        resolt_rad=resolt_rad+cmath.exp(complex(0, data_mod_2pi_right_sorted[i]))
    resolt_rad=resolt_rad/len(data_mod_2pi_right_sorted)
    #print(resolt_rad)
    if resolt_rad.real<0:
        resolt_rad=cmath.atan(resolt_rad.imag/resolt_rad.real)#resolt_degree=resolt_rad*180/math.pi
        average_angels[2]=resolt_rad.real+math.pi
    else:
        resolt_rad=cmath.atan(resolt_rad.imag/resolt_rad.real)#resolt_degree=resolt_rad*180/math.pi
        average_angels[2]=resolt_rad.real
    #end=timeforloop()
    average_angels[3]=average_angels[0]
    average_values[0]=left_sync[step]          
    average_values[1]=mid_sync[step] 
    average_values[2]=right_sync[step]
    average_values[3]=average_values[0]
    #####################################################################################
    #logaritmic for bar
    for i in range(0,len(values)):
        values[i]=math.log10((10*values[i])+1)
    for i in range(0,len(values_Righ)):
        values_Righ[i]=math.log10((10*values_Righ[i])+1)
    for i in range(0,len(values_left)):
        values_left[i]=math.log10((10*values_left[i])+1)    
    #####################################################################################
    ax4.set_thetagrids(angel, fmt="%.2f")
    plt.gcf().set_size_inches(6, 6)
    ax4.bar(theta, values_Righ, width=width, bottom=0.0, color="#0000ff", alpha=0.5)
    ax4.bar(theta, values_left, width=width, bottom=0.0, color="#00ff00", alpha=0.5)
    ax4.bar(theta, values, width=width, bottom=0.0, color="r", alpha=0.5)
    ax4.scatter(average_angels, average_values, c=average_colors, cmap='brg')
    ax4.plot(average_angels, average_values, color='k', linewidth=1) #khat
    ax4.fill(average_angels, average_values, color='k', alpha=0.25); #masahat
    ax4.set_xlim([0, 2*math.pi]),ax4.set_ylim([0, 1])
    #ax4.axes.get_yaxis().set_ticks([])
    del resolt_rad
    del resolt_degree   
#****************************************************************************************************                +--------+
#****************************************************************        Syncronyzation      ********                + Define +
#****************************************************************************************************                +--------+
def plot_Syncronyzation(ax5):  
    ax5.plot(Number_total, Total_sync, 'k', label='Total')
    ax5.plot(Number_total, left_sync, color="#00ff00", label='Left Part')
    ax5.plot(Number_total, mid_sync, color="r", label='Middle Part')
    ax5.plot(Number_total, right_sync, color="#0000ff", label='Right Part')
    ax5.axvline(x=step/100,color = "r", linestyle='dashed',linewidth=1, label='time='+str(step/100))
    ax5.set(xlabel="time (t)", ylabel="Synconey (r)")
    ax5.set_xlim([0, 20])
    ax5.set_ylim([0, 1])
    ax5.legend(loc=(-0.31,0.34))
#****************************************************************************************************                +--------+
#****************************************************************          Omega Arenge      ********                + Define +
#****************************************************************************************************                +--------+
def plot_Omega(ax6):  
    ax6.set(xlabel='Number of Node (i)',
        ylabel='Natural Frequency Layer2 (w)')
    ax6.scatter(arr_xomega_left, arr_yomega_left, s=6, color="#00ff00", label="Left Part Layer 2")#"#00ff00"
    ax6.scatter(arr_xomega_mid, arr_yomega_mid, s=6, color="r", label="Mid Part Layer 2")#"#00ff00"
    ax6.scatter(arr_xomega_right, arr_yomega_right, s=6, color="#0000ff", label="Right Part Layer 2")#"#00ff00"
    ax6.plot(number_omega, omega,linewidth=0.9, linestyle='dashed', color="k", label="Layer 1")#"#00ff00"
    ax6.legend(loc=(1.06,0.4)); # upper left corner
#****************************************************************************************************                +--------+
#**********************************************************     Color cyrcle Phase Connection   *****                + Define +
#****************************************************************************************************                +--------+
def plot_color_cyrcle_phase(ax7):  
    nx.draw(G,
            pos, 
            node_color=Color_of_node, 
            node_size=200, 
            alpha=0.4, 
            cmap=Color_map_hsv,
            ax=ax7)
    nx.draw_networkx_edges(nx.random_geometric_graph(14, 1, seed=896803), pos=nx.circular_layout(nx.random_geometric_graph(14, 1, seed=896803)), alpha=0.6,ax=ax7)
    plt.text(0.4, 1.03, '[1,277]', transform=ax7.transAxes, fontsize=12).set_bbox(dict(facecolor="#00ff00", alpha=0.5, edgecolor="#00ff00"))
    plt.text(0.36, -0.05, '[724,1000]', transform=ax7.transAxes, fontsize=12).set_bbox(dict(facecolor="#0000ff", alpha=0.5, edgecolor="#0000ff"))
    plt.gcf().colorbar(plt.cm.ScalarMappable(cmap=Color_map_hsv, norm=mpl.colors.Normalize(vmin=0, vmax=2*math.pi)), cax=plt.axes([0.925, 0.72, 0.01, 0.2]), ticks=np.linspace(0, 2*math.pi, 9)).set_label('Phase')   
    draw_self_loop(center=(.0, .0), radius=0.85, facecolor="#00ff00", edgecolor='gray', theta1=0, theta2=100)#"#0000ff"
    draw_self_loop(center=(.0, .0), radius=0.85, facecolor="r", edgecolor='gray', theta1=100, theta2=260)#"#0000ff"
    draw_self_loop(center=(.0, .0), radius=0.85, facecolor="#0000ff", edgecolor='gray', theta1=260, theta2=360)#"#0000ff"

#****************************************************************************************************                +--------+
#**********************************************************     Color cyrcle Phase Connection   *****                + Define +
#****************************************************************************************************                +--------+
def plot_Frecuency(ax8):      
    ax8.plot(timeforloop_Fre, f_total, color="k", label='Total')        
    ax8.plot(timeforloop_Fre, f_mid, color="r", label='Middle Part')     
    ax8.plot(timeforloop_Fre, f_left, color="#00ff00", label='Left Part')        
    ax8.plot(timeforloop_Fre, f_right, color="#0000ff", label='Right Part')
    ax8.set(xlabel="time (t)", ylabel="Average frequency (Hz)")
    ax8.axvline(x=step/100,color = "r", linestyle='dashed',linewidth=1, label='time='+str(step/100))
    ax8.set_xlim([0, 20])
    ax8.set_ylim([-14, 14])
    ax8.legend(loc=(-0.31,0.34))
#****************************************************************************************************                +--------+
#**********************************************************           coupling Syncroney        *****                + Define +
#****************************************************************************************************                +--------+   
def plot_coupling_Syncroney(ax9):        
    ax9.set(xlabel='Coupling (k)', ylabel='Synconey (r)')
    ax9.plot(timeforloop_Average_sync, Layer2_Average_sync, color="k",linewidth=2, label="Forward")#430303
    ax9.plot(timeforloop_Average_sync, Layer2_Average_sync_backward, color="r",linewidth=2, label="Backward")#430303
    ax9.axvline(x=(copling-0.0032) ,color = "r", linestyle=(0, (3, 5, 1, 5)),linewidth=2, label='Copling='+str(copling))
    ax9.legend(loc=(1.06,0.4)); # upper left corner
    ax9.set(xlim=[0, 3], ylim=[0, 1])
#****************************************************************************************************                +--------+
#**********************************************************             Corrolation             *****                + Define +
#****************************************************************************************************                +--------+   
def plot_Corrolation(ax10):           
    plt.gcf().colorbar(ax10.pcolormesh(Corolation,cmap='jet', vmin=-1, vmax=1), cax=plt.axes([0.925, 0.08, 0.01, 0.2]), ticks=np.linspace(-1,1, 9)).set_label('Correlation')
    ax10.set_ylabel('Number of Node (i)')
    ax10.set_xlabel('Number of Node (i)')
#****************************************************************************************************                +--------+
#**********************************************************        frequency samples            *****                + Define +
#****************************************************************************************************                +--------+      
def plot_frequency_samples(ax11): 
    #single
    
    ax11.plot(timeforloop_Fre, Frequency[499], color="#333333", label='i=500',alpha=1)

    
    ax11.plot(timeforloop_Fre, Frequency[199], color="#cc0066", label='i=200',alpha=0.9)
    #ax11.plot(timeforloop_Fre, Frequency[366], color="#A52A2A", label='i=367')
    #ax11.plot(timeforloop_Fre, Frequency[455], color="#CD3333", label='i=456')
    #ax11.plot(timeforloop_Fre, Frequency[544], color="#EE3B3B", label='i=545')
    #ax11.plot(timeforloop_Fre, Frequency[633], color="#FF4040", label='i=634')
    #ax11.plot(timeforloop_Fre, Frequency[722], color="#FF6A6A", label='i=723')

    #ax11.plot(timeforloop_Fre, Frequency[69], color="#008000", label='i=70')
    #ax11.plot(timeforloop_Fre, Frequency[138], color="#228B22", label='i=139')
    #ax11.plot(timeforloop_Fre, Frequency[208], color="#32CD32", label='i=209')
    #ax11.plot(timeforloop_Fre, Frequency[276], color="#90EE90", label='i=276')


    ax11.plot(timeforloop_Fre, Frequency[799], color="#ff6600", label='i=800',alpha=0.8)
    #ax11.plot(timeforloop_Fre, Frequency[792], color="#48D1CC", label='i=793')
    #ax11.plot(timeforloop_Fre, Frequency[861], color="#20B2AA", label='i=862')
    #ax11.plot(timeforloop_Fre, Frequency[930], color="#008B8B", label='i=931')
    #ax11.plot(timeforloop_Fre, Frequency[999], color="#008080", label='i=1000')

    ax11.set(xlabel="time (t)", ylabel="Average frequency (Hz)")
    ax11.axvline(x=step/100,color = "r", linestyle='dashed',linewidth=1, label='time='+str(step/100))
    ax11.set_xlim([0, 20])
    ax11.set_ylim([-14, 14])
    ax11.legend(loc=(-0.31,0.34))
#****************************************************************************************************                +--------+
#**********************************************************        frequency total              *****                + Define +
#****************************************************************************************************                +--------+      
def plot_frequency_total(ax12): 
    ax12.axvline(x=step/100,color = "r", linestyle='dashed',linewidth=1, label='time='+str(step/100))
    for timeforloop in range(dw_left_cut,dw_Right_cut):#for timeforloop in range(dw_left_cut,dw_Right_cut):
        ax12.plot(timeforloop_Fre, Frequency[timeforloop], color="r", label='Total Synconey',alpha=0.03)  
    for timeforloop in range(0,dw_left_cut):#for timeforloop in range(0,dw_left_cut):
        ax12.plot(timeforloop_Fre, Frequency[timeforloop], color="#00ff00", label='Total Synconey',alpha=0.08)
    for timeforloop in range(dw_Right_cut,Number_of_node):#for timeforloop in range(dw_Right_cut,Number_of_node):
        ax12.plot(timeforloop_Fre, Frequency[timeforloop], color="#0000ff", label='Total Synconey',alpha=0.08)
    ax12.set(xlabel="time (t)", ylabel="Average frequency (Hz)")
    ax12.set_xlim([0, 20])
    ax12.set_ylim([-14, 14])
    #ax12.legend.remove()
#################################################################################################                    +--------+
##########                                                                             ##########                    +        +
##########                                 PLOTS                                       ##########                    +  PLOTS +
##########                                                                             ##########                    +        +
#################################################################################################                    +--------+
#for iii in range(0,10):
start_time=time.time()
#define plot 
plt.close('all')
fig = plt.figure()
ax1 = plt.subplot(4,4,7,projection="polar")#Left Phase circle
ax2 = plt.subplot(4,4,11,projection="polar")
ax3 = plt.subplot(4,4,15,projection="polar")
ax4 = plt.subplot(4,4,3,projection="polar")
ax5 = plt.subplot(4,4,(1,2))
ax6 = plt.subplot(4,4,12)
ax7 = plt.subplot(4,4,4)
ax8 = plt.subplot(4,4,(5,6))
ax9 = plt.subplot(4,4,8)
ax10= plt.subplot(4,4,16)
ax11= plt.subplot(4,4,(9,10))
ax12= plt.subplot(4,4,(13,14))

plot_phase_left(ax1)
plot_phase_mid(ax2)
plot_phase_Right(ax3)
plot_Distribution(ax4)
plot_Syncronyzation(ax5)
plot_Omega(ax6)
plot_color_cyrcle_phase(ax7)
plot_Frecuency(ax8)
plot_coupling_Syncroney(ax9)
plot_Corrolation(ax10)
plot_frequency_samples(ax11)
plot_frequency_total(ax12)
plt.suptitle("Seif-Algorithm, Forward increasing interlayer coupling σ="+str(copling)+" \nIntralayer coupling ι="+str(landa)+", Δω="+str(dW)+", Intralayer Frustration α="+str(Degree)+", time=" +str(step/100), fontdict=font1)
plt.subplots_adjust(top = 0.9, bottom=0.1, hspace=0.3, wspace=0.4)
plt.gcf().set_size_inches(21, 19)
try:
    os.mkdir(path)
except OSError:
    pass
plt.savefig(path+str(step)+".png", dpi=200)
del ax1
del ax2
del ax3
del ax4
del ax5
del ax6
del ax7
del ax8
del ax9
del ax10
del ax11
del ax12
plt.close(fig)
fig.clear()
fig.clf()
gc.collect()
print("Time: {:.3f} sec".format(time.time()-start_time))
print('Ram usage: {:.2f} MB'.format(process.memory_info().rss/1000000))


```
