#Script to calculate the GTensor from given DAXM Data

#!/usr/bin/env python 
import numpy as np
import math
import os
import argparse

from scipy import spatial
from scipy.interpolate import Rbf

from numpy import linalg as La
from CurlCalc import CurlCalc

os.system('clear')

parser = argparse.ArgumentParser()
parser.add_argument("--ascii", help=" Enter the .txt filename")
args=parser.parse_args()

# Header Count on the ASCII File
A=open(args.ascii,'r')
HeadStr=A.readline()
AsciiHead=int(HeadStr.split()[0])

P=[]
for i in range(AsciiHead):
  P.append(A.readline())
P[-1]=P[-1].split('\n')[0]+'\tG11\tG12\tG13\tG21\tG22\tG23\tG31\tG32\tG33\tGNorm\n'
# List containing the names of the data items
VarName=P[-1].split() 

A.close()

# Get Starting Index Values for the data items
for ii in range(len(VarName)):
  if VarName[ii]=='1_APS':
    CoordIndex00=ii
  if VarName[ii]=='1_aStar':
    Recip_a=ii
  if VarName[ii]=='1_bStar':
    Recip_b=ii
  if VarName[ii]=='1_cStar':
    Recip_c=ii
  if VarName[ii]=='1_euler':
    EulerIndex00=ii


# Calculate BIdeal
a,b,c=0.2950800,0.2950800,0.4685500, # Lattice Parameters for Titanium

alpha,beta,gamma=np.pi/2,np.pi/2,2*np.pi/3

Ca,Cb,Cg=np.cos(alpha),np.cos(beta),np.cos(gamma)
Sg=np.sin(gamma)

V=(a*b*c)*np.sqrt(1+2*Ca*Cb*Cg-Ca**2-Cb**2-Cg**2) # Volume of Unit Cell

# Real Space Lattice Vectors for a strain-free, rotation-free unit cell. a is chosen coincident with the X-axis


aR,bR=np.array([a,0.,0.]), np.array([b*Cg,b*Sg,0.])
cR=np.array([c*Cb,c*((Ca-Cb*Cg)/Sg),V/(a*b*Sg)])

# Normalize The Lattice Parameters
aR,bR,cR=aR/np.linalg.norm(aR),bR/np.linalg.norm(bR),cR/np.linalg.norm(cR)

BIdeal=np.vstack((aR,bR,cR))#Ideal Real Space Lattice Vector for Strain Free Unit Cell

#BIdeal=np.transpose(BIdeal)

# Function to Calculate Euclidean Distance between a chosen point and other points in the cloud
def Euclid(V0,V1): # V0--> Chosen Point, V1--> Array of points
# Evaluate Euclidean distance find nearest points!
 K=np.zeros((len(V1),))
 #for ii in range(len(V1)):
 K=np.sqrt((V0[0]-V1[:,0])**2+(V0[1]-V1[:,1])**2+(V0[2]-V1[:,2])**2) 
 return K

def UnitVec(R):
 R=np.asarray([R[j]/La.norm(R[j]) for j in range(len(R))])
 return R


# Extract Data from ASCII Table:


F=np.genfromtxt(args.ascii, skip_header=AsciiHead+1)


Curl_Fe_inv=np.zeros((len(F),3,3)) # Initialize Array to Store Curl of Fe_inv 
G=np.zeros((len(F),3,3))
GNorm=np.zeros((len(F),1))

CC=CurlCalc(F[:,CoordIndex00:CoordIndex00+3],F[:,Recip_a:Recip_a+3],\
            F[:,Recip_b:Recip_b+3],F[:,Recip_c:Recip_c+3])

Real=CC.RealSpace() # Real Space Vectors

Fe=CC.DefGrad(BIdeal,Real)   # Elastic Deformation Gradient
Fe_inv=np.empty((len(Fe),3,3))  # Initialize array for inverse of the elastic deformation gradient


for i in range(len(Fe)):
  Fe_inv[i]=La.inv(Fe[i]) 

ED=20 # Search Radius in microns (um)
tree=spatial.cKDTree(F[:,:CoordIndex00+3])
for ii in range(len(F)):
  A=tree.query_ball_point(F[ii,:CoordIndex00+3],ED)
  W=np.zeros((len(A),6))
  W[:,0],W[:,1]=A, Euclid(F[ii,:CoordIndex00+3],F[A,:CoordIndex00+3])
  W[:,2],W[:,3],W[:,4]=F[A,EulerIndex00],F[A,EulerIndex00+1],F[A,EulerIndex00+2]
  W=W[np.argsort(W[:,1])]
  W[:,5]=CC.MisArray(W[0,2:5],W[:,2:5])
  W=W[W[:,5]<=5]  # Using Misorientation Filter of 5-degrees: Comment out to turn off!
  if len(W)>=10: # Minimum number of 9 points for Radial Basis Function Interpolation
    #Curl_Fe_inv[ii]=np.zeros((3,3))
  #else:
    
    ID=W[:,0].astype(int)
    R=np.longdouble(UnitVec(F[ID,:CoordIndex00+3]))     # Increase precision to Quadruple Floating point format to avoid division by zero  

    #Evaluate Curl of Inverse of Deformation Gradient
    Curl_Fe_inv[ii]=CC.CurlTens(Fe_inv[ID],R,2e-4,2e-4,2e-4) #(self,T,X,delX,delY,delZ)


  G[ii]=np.linalg.det(Fe[ii])*np.dot(Fe_inv[ii],Curl_Fe_inv[ii])*1e12 
  GNorm[ii]=La.norm(G[ii])

G=G.ravel().reshape(len(F),9)


# Write Data to the output file
OutFile=args.ascii.split('.')
OutFile=OutFile[0]+'_'+'GTensor'+'.txt'
outfile=open(OutFile,'w')
outfile.write(HeadStr)
for item in P:
  outfile.write(item)
np.savetxt(outfile,np.concatenate((F,G,GNorm),axis=1),fmt='%7.5f')
outfile.close()
