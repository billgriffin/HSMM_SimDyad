#couple_types = ["High","Med","Low"]
data_groups = ["Cluster1","Cluster2","Cluster3","Cluster4","Cluster5"]

file_dict = {}
gamma_dict = {}
pgaussian_dict = {}

file_dict["Cluster1"] = {"Cluster1": "[203,212,217,225]"}
file_dict["Cluster4"] = {"Cluster4": "[200,202, 213]"}
file_dict["Cluster2"] = {"Cluster2": "[204,205,208,209,210,211,214,215,219,224]"}
file_dict["Cluster3"] = {"Cluster3": "[206,207,218,220,222,228,230]"}
file_dict["Cluster5"] = {"Cluster5": "[201,221,223,226,227,229]"}

#gamma_dict["High"] = {}
gamma_dict["Cluster1"] = ([2.76875114, 2.85382056])
gamma_dict["Cluster4"] = ([1.42923284, 1.72062993])

#gamma_dict["Med"] = {}
gamma_dict["Cluster2"] = ([3.15076423, 3.00486636])
gamma_dict["Cluster3"] = ([3.65220284,  3.9106607])

#gamma_dict["Low"] = {}
gamma_dict["Cluster5"]  = ([3.21069598, 4.04306316])

#pgaussian_dict = {}
pgaussian_dict["Cluster1"] = ([[1.45271406e-04, -3.36416997e-05],[ -3.36416997e-05, 1.92515130e-04]])
pgaussian_dict["Cluster4"] = ([[0.00094602, 0.0001773 ], [ 0.0001773, 0.00078666]])

#pgaussian_dict = {}
pgaussian_dict["Cluster2"]  = ([[1.41064331e-04,  -2.32959192e-0],[ -2.32959192e-05,  1.29842141e-04]])
pgaussian_dict["Cluster3"]  = ([[1.44527265e-04, -4.36093796e-05],[ -4.36093796e-05,  1.57100483e-04]])

#pgaussian_dict = {}
pgaussian_dict["Cluster5"]  = ([[2.96013342e-04,  -2.41472590e-0 ], [ -2.41472590e-05,   1.86607416e-04]])


count = 1
#for couple_type in couple_types:
for data_group in data_groups:
   files = file_dict[data_group]
   # resampling
   n_resample_group = [10000]
   for n_resample in n_resample_group:
      # hyper-hyper parameters
      alpha_group = [4.0,8.0]
      for alpha in alpha_group:
         gamma_group = [4.0,8.0]
         for gamma in gamma_group:
            # hyper-gaussian
            mu = pgaussian_dict[data_group][0]
            lmda = pgaussian_dict[data_group][1]
            kappa = 0.25
            nu = 4
            # hyper-possion
            k,theta = gamma_dict[data_group]

            # number states
            n_states_group = [15,20]
            for n_states in n_states_group:
               # number trunc
               n_trunc_group = [5,10]
               for n_trunc in n_trunc_group:
                  #
                  # start writing
                  file_name = 'parameter%d.conf' % count
                  o = open(file_name, 'w')
                  #o.write(couple_type)
                  o.write('All') #
                  o.write("\n")
                  o.write(data_group)
                  o.write("\n")
                  o.write(file_dict[data_group][data_group])
                  #o.write(files[1])
                  o.write("\n")
                  o.write(str(n_resample))
                  o.write("\n")
                  o.write(str(alpha))
                  o.write("\n")
                  o.write(str(gamma))
                  o.write("\n")
                  o.write("np.array(%s)"%mu)
                  o.write("\n")
                  o.write("np.array(%s)"%lmda)
                  o.write("\n")
                  o.write(str(kappa))
                  o.write("\n")
                  o.write(str(nu))
                  o.write("\n")
                  o.write(str(k))
                  o.write("\n")
                  o.write(str(theta))
                  o.write("\n")
                  o.write(str(n_states))
                  o.write("\n")
                  o.write(str(n_trunc))
                  o.close()
                  count += 1