inFile = open("u1.base", "r")
lines = inFile.readlines()
inFile.close()

usersR = {}

for i in range(0,len(lines)):
	line = lines[i].split("\t")
	itemR = {}
	itemR[line[1]] = line[2]
	usersR.setdefault(line[0],[]).append(itemR)


for i in range(0,len(usersR)):
	user = usersR[str(i+1)]
	minR = 5
	maxR = 0
	for j in range(0,len(user)):
		item = user[j]
		minR = min(minR,int(item.values()[0]))
		maxR = max(maxR,int(item.values()[0]))
	for j in range(0,len(user)):
		item = user[j]
		val = format((int(item.values()[0])-minR)/float(maxR-minR),'.2f')
		item[item.keys()[0]] = str(val)


outFile = open("user.data","wb")

for i in range(0,len(lines)):
	line = lines[i].split("\t")
	for j in range(0,len(usersR[line[0]])):
		item = usersR[line[0]][j]
		if item.keys()[0]==line[1]:
			line[2] = item.values()[0]
			outFile.write('\t'.join(line))
			break
outFile.close()
outFile = open("user.data","r")
lines = outFile.readlines()
print len(lines)