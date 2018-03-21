import re
import string
import os
import struct
import numpy as np
# get equal numbers of Positive Samples and Negtive Samples

def getSubBlock(cuData, cuSize):
	t0 = 0
	t1 = 0
	t2 = 0
	t3 = 0

	total_num = cuSize * cuSize
	half_num  = total_num / 2
	half_size = cuSize / 2

	sub0 = np.arange(0, half_size * half_size)
	sub1 = np.arange(0, half_size * half_size)
	sub2 = np.arange(0, half_size * half_size)
	sub3 = np.arange(0, half_size * half_size)

	k = 0
	while k < half_num:
		i = 0
		while i < half_size:
			sub0[t0] = cuData[k]
			t0 += 1
			k += 1
			i += 1
		while i < cuSize:
			sub1[t1] = cuData[k]
			t1 += 1
			k += 1
			i += 1
	while k < total_num:
		i = 0
		while i < half_size:
			sub2[t2] = cuData[k]
			t2 += 1
			k += 1
			i += 1
		while i < cuSize:
			sub3[t3] = cuData[k]
			t3 += 1
			k += 1
			i += 1
	return sub0, sub1, sub2, sub3
def getSCCD(cuData, size):
	sub0, sub1, sub2, sub3 = getSubBlock(cuData, size)
	var0 = sub0.var()
	var1 = sub1.var()
	var2 = sub2.var()
	var3 = sub3.var()

	var_mean = (var0 + var1 + var1 + var3) / 4

	SCCD0 = (var0 - var_mean) **2
	SCCD1 = (var1 - var_mean) **2
	SCCD2 = (var2 - var_mean) **2
	SCCD3 = (var3 - var_mean) **2
	SCCD  = (SCCD0 + SCCD1 + SCCD1 + SCCD3) / 4

	return SCCD

def calBlockFlatness(cuData):
	i = 0
	temp = np.zeros(cuData.size, dtype=np.int64)
	while i < cuData.size:
		a = int(cuData[i])
		temp[i] = int(a*a)
		i += 1
	t1 = float(cuData.sum()) / temp.sum()
	t2 = float(cuData.sum()) / cuData.size
	bf = t1 * t2
	return bf

def getCuData(cuData, label):
	i = 0
	chunk = ""
	while i < cuData.size:
		chunk += str(cuData[i])
		chunk += "  "
		i += 1
	chunk += str(label) + "\n"
	return chunk


def parseData(readData, writelabel, dirWritePicPath, size):
	num = 100000
	j = 1
	i = 0
	samplePos = 16000;
	sampleNeg = 8000;
	while samplePos > 0 or sampleNeg > 0:
		label = readData.read(1)
		if label != b'\x01' and label != b'\x00':
			print("label Error!")
			break
		if label == "":
			print("End Processing!")
			break

		if label == b'\x01':
			if samplePos > 0:
				label = 1
				samplePos -= 1
			else:
				uselessData = readData.read(size * size)
				continue
		if label == b'\x00':
			if sampleNeg > 0:
				label = 0
				sampleNeg -= 1
			else:
				uselessData = readData.read(size * size)
				continue

		cuData = readData.read(size * size)
		cuData = np.frombuffer(cuData, dtype=np.uint8) # cuData = 0 ~ 255

		SCCD = getSCCD(cuData, size)

		cuMean = cuData.mean()

		cuVar =  cuData.var()

		cuBF  =  calBlockFlatness(cuData)

		#chunk = str(cuMean) + "  " + str(cuVar) + "  " + str(cuBF) + "  " + str(label) + "\n"
		chunk = str(cuVar) + "  " + str(cuBF) + "  " + str(label) + "\n"
		#chunk = str(SCCD) + "  " + str(cuVar) + "  " + str(cuBF) + "  " + str(label) + "\n"
		#chunk = str(SCCD) + "  " + str(cuBF) + "  " + str(label) + "\n"
		#chunk = str(SCCD) + "  " + str(cuVar) +  "  " + str(label) + "\n"

		#chunk = getCuData(cuData, label)

		writelabel.write(chunk)

		i = i + 1
	print(samplePos)
	print(sampleNeg)

def parseSet(dirReadData, seqName, cuSize):
    readData = open(dirReadData, 'rb')
    dirWritelabel = '/Users/mengwang/Documents/MyCode/Machine Learning/cuSplit/' + seqName + '_labels.data'
    writelabel = open(dirWritelabel, 'w')
    dirWriteDataPath = '/Users/mengwang/Documents/MyCode/Machine Learning/cuSplit/'+ seqName + '/'

    parseData(readData, writelabel, dirWriteDataPath, cuSize)
    readData.close()
    writelabel.close()

def parseWrapper(cuSize, qp, dataSet):
    print("Extracting CU"+ str(cuSize) + "_QP" + str(qp)+ "_" + dataSet + "...")
    dirReadData = '/Users/mengwang/Documents/MyCode/Machine Learning/cuSplit/data/CU' + str(cuSize) + 'Samples_AI_CPIH_768_1536_2880_4928_qp' + str(qp) + '_' + dataSet + '.dat'
    seqName='CU' + str(cuSize) + '_QP' + str(qp) + '_' + dataSet
    parseSet(dirReadData, seqName, cuSize)


if __name__=="__main__":
	#parseWrapper(16, 22, "Valid")
	#parseWrapper(16, 22, "Test")

	#parseWrapper(32, 22, "Valid")
	#parseWrapper(32, 22, "Test")

	parseWrapper(64, 22, "Valid")
	parseWrapper(64, 27, "Test")
