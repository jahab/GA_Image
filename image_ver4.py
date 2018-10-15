import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import random
import math
from operator import itemgetter
import multiprocessing

#pool=multiprocessing.Pool(processes=2)

#my_image=Image.open("picasso_bull_plate_3(1).jpg")
my_image=Image.open("red.jpg")
pixels=my_image.convert('RGBA')
Pix_val_target=pd.Series(list(my_image.getdata()))
f=lambda i: np.array(i)
Pix_val_target=Pix_val_target.apply(f)

Image_size=my_image.size

#create Population
PopMax=50
pop_poly=10
RGBA=8+8+8+8
##for ellipses
x1=10
y1=10
x2=10
y2=10
k=10
####
#x3
#y3
bitsize=RGBA+x1+y1+x2+y2

def create_init_pop(pop_poly):
    arr=pd.Series(np.arange(0,pop_poly))
    j=lambda i:np.random.binomial(1,0.5,bitsize)
    arr=arr.apply(j)
    return arr
####

def bintod_cordinates(binary_number):
    Ux=max(Image_size)
    Lx=0
    Uy=min(Image_size)
    Ly=0
    k=10
    #print(U,L)
    Ua=20
    La=150
    summ=0
    for i in range(32,32+k):
        summ=summ + (2**(i-32))*binary_number[i]
    ####
    x1= (Lx + ((Ux-Lx)/((2**k)-1))*summ).astype(int)
    summ=0
    for i in range(32+k,32+2*k):
        summ=summ + (2**(i-(32+k)))*binary_number[i]
    ####
    y1= (Ly + ((Uy-Ly)/((2**k)-1))*summ).astype(int)
    summ=0
    for i in range(32+2*k,32+3*k):
        summ=summ + (2**(i-(32+2*k)))*binary_number[i]
    ####
    x2= (Lx + ((Ux-Lx)/((2**k)-1))*summ).astype(int)
    summ=0
    for i in range(32+3*k,32+4*k):
        summ=summ + (2**(i-(32+3*k)))*binary_number[i]
    ####
    y2= int(Ly + ((Uy-Ly)/((2**k)-1))*summ)
    return ((x1,y1),(x2,y2))
######

def bintod_RGB(binary_number):
    U=255
    L=0
    Ua=20
    La=150
    summ=0
    for i in range(0,8):
        summ=summ + (2**i)*binary_number[i]
    ####
    R= (L + ((U-L)/((2**8)-1))*summ).astype(int)
    summ=0
    for i in range(8,16):
        summ=summ + (2**(i-8))*binary_number[i]
    ####
    G= (L + ((U-L)/((2**8)-1))*summ).astype(int)
    summ=0
    for i in range(16,24):
        summ=summ + (2**(i-16))*binary_number[i]
    ####
    B= (L + ((U-L)/((2**8)-1))*summ).astype(int)
    summ=0
    for i in range(24,32):
        summ=summ + (2**(i-24))*binary_number[i]
    ####
    A= (La + ((Ua-La)/((2**8)-1))*summ).astype(int)
    return (R,G,B,A)
######

def create_image(pops):
    pixel_cluster_of_Image=pd.DataFrame()
    binary_images=[]
    for k in range(len(pops)):
        #pops=create_init_pop(pop_poly)
        binary_images.append(pops[k])

        img = Image.new('RGB', Image_size)
        #make list of triangle coordinates
        #create polygons on the black box image
        drw = ImageDraw.Draw(img, 'RGBA')
        for i in range(len(pops[k])):
            #rgba=(np.random.randint(0,255),np.random.randint(0,255), np.random.randint(0,255),np.random.randint(20,150))
            rgba=bintod_RGB(pops[k][i])
            #rgba=(np.random.randint(0,255),np.random.randint(0,255), np.random.randint(0,255),np.random.randint(20,150))

            alpha=(np.random.randint(Image_size[0]),np.random.randint(Image_size[1]))
            #cordinate_triangle=((np.random.randint(Image_size[0]),np.random.randint(Image_size[1])), (np.random.randint(Image_size[0]),np.random.randint(Image_size[1])), (np.random.randint(Image_size[0]),np.random.randint(Image_size[1])),alpha)
            #cordinate_ellipse=((np.random.randint(Image_size[0]),np.random.randint(Image_size[1])), (np.random.randint(Image_size[0]),np.random.randint(Image_size[1])))
            cordinate_ellipse=bintod_cordinates(pops[k][i])
            #cordinate_ellipse=((np.random.randint(Image_size[0]),np.random.randint(Image_size[1])), (np.random.randint(Image_size[0]),np.random.randint(Image_size[1])))

            #print(cordinate_ellipse)

            drw.ellipse(cordinate_ellipse,rgba)
        ###
        pixel_cluster_of_Image=pd.concat([pixel_cluster_of_Image,pd.Series(list(img.getdata())).apply(f)],axis=1)
        #print(list(img.getdata()))
        del drw
        #img.show()
    return (pixel_cluster_of_Image,binary_images)
#####

def fitness_func(Pix_val_target,pixel_cluster_of_Image):
    total_fitness=pd.Series(np.arange(0,len(pixel_cluster_of_Image.columns)))
    #print(total_fitness)

    ff=lambda k: np.sqrt(sum(sum((pixel_cluster_of_Image.iloc[:,k]-Pix_val_target)**2)))

    total_finness=total_fitness.apply(ff)

    for i in range(0,len(pixel_cluster_of_Image.columns)):
        fitness=np.sqrt(sum(sum((pixel_cluster_of_Image.iloc[:,i]-Pix_val_target)**2)))
        total_fitness=total_fitness.set_value(i,fitness)

    return(total_finness)
####

def tournament_selection(ranks,images_cluster, bin_cluster):
    #select randomly two individuals
    selected_pool=pd.DataFrame()
    selected_binary_pool=[]
    for i in range(len(ranks)):
        choice=random.sample(range(len(ranks)),2)

        ran_choice=np.where(ranks==min(ranks[choice]))[0]
        #print(ran_choice)
        selected_pool=pd.concat([selected_pool,images_cluster.iloc[:,ran_choice]],axis=1)
        selected_binary_pool.append(itemgetter(*ran_choice)(bin_cluster))
        #print(selected_pool)
    return(selected_pool,selected_binary_pool)
####
"""
def convert_to_binary(x,y):
    gx=list(''.join(np.repeat('0',8)))
    #cx=pd.Series(np.tile(gx,(len(popX),1)).flatten())
    binx=bin(x)[2:]
    gx[8-len(binx):]=binx
    gy=list(''.join(np.repeat('0',8)))
    #cx=pd.Series(np.tile(gx,(len(popX),1)).flatten())
    biny=bin(y)[2:]
    gy[8-len(biny):]=biny
    return (gx,gy)
#####
"""

def crossover(pool):
    updatedpool=[]
    for i in range(0,PopMax):
        child1=pd.Series()
        child2=pd.Series()
        total_ch=pd.Series()
        choice=random.sample(range(PopMax),2)
        rand_slct_pool=itemgetter(*choice)(pool)
        for j in range(0,len(pool[i])):
            fixnumber= np.random.randint(0,bitsize)
            tempx1=rand_slct_pool[0][j][fixnumber:]
            tempx2=rand_slct_pool[1][j][fixnumber:]
            child1x=np.append(rand_slct_pool[1][j][:fixnumber],tempx1)
            child2x=np.append(rand_slct_pool[0][j][:fixnumber],tempx2)
            #child1=child1.append(child1x)
            #child=child.append(child2x)
            child1=child1.set_value(j,child1x)
            child2=child2.set_value(j,child2x)
        #total_ch=total_ch.append(child1)
        #total_ch=total_ch.append(child2, ignore_index=True)
        updatedpool.append(child1)
        updatedpool.append(child2)


    return updatedpool
####

def mutation(crossoverpops,mu_rate):
    for i in range(len(crossoverpops)):
        for j in range(len(crossoverpops[i])):
            for k in range(len(crossoverpops[i][j])):
                if np.random.sample()<mu_rate:
                    crossoverpops[i][j][k]=abs(crossoverpops[i][j][k]-1)
    return crossoverpops
####

def show_image(pops):
    img = Image.new('RGB', Image_size)
    #make list of triangle coordinates
    #create polygons on the black box image
    drw = ImageDraw.Draw(img, 'RGBA')
    for i in range(len(pops)):
        #rgba=(np.random.randint(0,255),np.random.randint(0,255), np.random.randint(0,255),np.random.randint(20,150))
        rgba=bintod_RGB(pops[i])
        #rgba=(np.random.randint(0,255),np.random.randint(0,255), np.random.randint(0,255),np.random.randint(20,150))

        alpha=(np.random.randint(Image_size[0]),np.random.randint(Image_size[1]))
        cordinate_ellipse=bintod_cordinates(pops[i])
        drw.ellipse(cordinate_ellipse,rgba)
    ###
    #pixel_cluster_of_Image=pd.concat([pixel_cluster_of_Image,pd.Series(list(img.getdata())).apply(f)],axis=1)
    #print(list(img.getdata()))
    del drw
    d=img.show()
    #d.quit()


######

"""
def selection_for_crossover(selected_pool):
    for i in range(len(selected_pool)):
        for j in range(3):
            crossover(selected_pool.iloc[:,0][i][j],selected_pool.iloc[:,1][i][j])

####
"""

#def elitism():



####
def fit_select(pops):
    #find the RGB images of the new cluster
    image_cluster=create_image(pops)
    print(len(image_cluster[0].columns))
    updated=[]

    #sort the ranks in the ascending order and take the fitest Image
    ranksofindex=(fitness_func(Pix_val_target,image_cluster[0]).sort_values().index)[0:PopMax]
    for i in range(len(pops)):
        if i in ranksofindex:
            updated.append(pops[i])

    return updated

####

#create initial population

#####
#create image from Binary
#for i in range(0,100):


init_pop_cluster=[]
for i in range(PopMax):
    pops=create_init_pop(pop_poly)
    init_pop_cluster.append(pops)

while(True):
    images_cluster=create_image(init_pop_cluster)

    #disp_image=show_image(init_pop_cluster[1])

    RGB_of_Images=images_cluster[0] #Numeric RGB combinations of the Population

    binary_encoded_images=images_cluster[1] #Encoded  RGB and polygons of the Population

    ranks=fitness_func(Pix_val_target,RGB_of_Images)
    print(ranks.sort_values().head(3))

    select_pool=tournament_selection(ranks,RGB_of_Images,binary_encoded_images)
    selected_bin_pool=select_pool[1]


    crossoverpops=crossover(selected_bin_pool)

    mu_rate=0.02
    mutate_pops=mutation(crossoverpops,mu_rate)

    #select fittest from crossover and mutation
    select_fit=fit_select(mutate_pops)
    init_pop_cluster=select_fit
###

#r= pool.map(run())
#pool.close()
