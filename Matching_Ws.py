def HisMatchedWs(x,y):
    for i in range(0,len(x)):
       for j in range(i,len(x)):
           if i!=j:
               if 120<(x[i]+x[j]).M() and (x[i]+x[j]).M()<130:
                   y.append(1)
               else: 
                   y.append(0)
