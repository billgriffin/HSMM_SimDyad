
couple_types = ["High","Med","Low"]
data_groups = ["All","Cluster1","Cluster2"]

# comment above 2 lines and uncomment below 2 lines to enable "All" data
#couple_types = ["All"]
#data_groups = ["All"]

file_dict = {}
gamma_dict = {}
pgaussian_dict = {}

file_dict["All"] = {"All": "[200,203,206,208,209,211,212,213,217,225, 202,205,207,210,214,215,218,219,220,228, 201,204,221,222,223,224,226,227,229,230]"}
file_dict["High"] = {"All": "[200,203,206,208,209,211,212,213,217,225]",
                     "Cluster1": "[200,203,206,208,211,212,213]",
                     "Cluster2": "[209,217,225]"}
file_dict["Med"]  = {"All": "[202,205,207,210,214,215,218,219,220,228]",
                     "Cluster1": "[202,214,215,218,220,228]",
                     "Cluster2": "[205,207,210,219]"}
file_dict["Low"]  = {"All": "[201,204,221,222,223,224,226,227,229,230]",
                     "Cluster1": "[201,204,222,223,226,227]",
                     "Cluster2": "[221,224,229,230]"}

gamma_dict["All"] = {"All": (0.41540261042,4.19266661584)}
gamma_dict["High"] = {}
gamma_dict["High"]["Cluster1"]=(0.472480370168,8.06389929734)
gamma_dict["High"]["Cluster2"]=(0.97141411998,1.01156014374)
gamma_dict["High"]["All"]= (0.327510903202,6.95309311019)
gamma_dict["Med"] = {}
gamma_dict["Med"]["Cluster1"] = (0.228628629288,20.4943004114)
gamma_dict["Med"]["Cluster2"] = (0.405961550726,5.34337175978)
gamma_dict["Med"]["All"] = (1.79849793392,2.8410033299)
gamma_dict["Low"] = {}
gamma_dict["Low"]["Cluster1"] = (0.803258959346,4.11636855485)
gamma_dict["Low"]["Cluster2"] =  (0.404820922675,2.1497882152)
gamma_dict["Low"]["All"] = (1.60817852754,1.54314550502)


pgaussian_dict["All"] = {}
pgaussian_dict["All"]["All"] = ([4.09819937,4.47678757],[[6.50687070e-06,-2.39446035e-06],[-2.39446035e-06, 5.69199119e-06]],43056.0)
pgaussian_dict["High"] = {}
pgaussian_dict["High"]["Cluster1"] = ([3.16951466,3.43251586],[[4.99158996e-05,-2.48953802e-05],[-2.48953802e-05, 4.96568136e-05]],5038.0)
pgaussian_dict["High"]["Cluster2"] = ([4.20916414,3.6126802 ],[[8.73342578e-05,-1.09223429e-05],[-1.09223429e-05, 1.37146868e-04]],2161.0)
pgaussian_dict["High"]["All"] = ([3.48256254,3.48755908],[[2.98945906e-05,-1.27972544e-05],[-1.27972544e-05, 3.48536960e-05]],7197.0)
pgaussian_dict["Med"] = {}
pgaussian_dict["Med"]["Cluster1"] = ([4.29091644,4.78586817],[[6.64177496e-05,-3.81490136e-05],[-3.81490136e-05, 7.66026424e-05]],4259.0)
pgaussian_dict["Med"]["Cluster2"] = ([3.96287155,4.09299707],[[1.04115978e-04,-2.43538234e-05],[-2.43538234e-05, 1.01289203e-04]],2882.0)
pgaussian_dict["Med"]["All"] = ([4.15967512,4.50749922],[[3.97097174e-05,-1.80099323e-05],[-1.80099323e-05, 4.22626254e-05]],7139.0)
pgaussian_dict["Low"] = {}
pgaussian_dict["Low"]["Cluster1"] = ([4.76226711,5.55416012],[[9.80025288e-05,-1.35156370e-05],[-1.35156370e-05, 6.54718096e-05]],4320.0)
pgaussian_dict["Low"]["Cluster2"] = ([4.47898436,5.24558115],[[1.90531588e-04,-5.69998404e-09],[-5.69998404e-09, 8.39533604e-05]],2879.0)
pgaussian_dict["Low"]["All"] = ([4.65027618,5.43225908],[[6.46839399e-05,-5.32961167e-06],[-5.32961167e-06, 3.67522807e-05]],7197.0)

count = 1
for couple_type in couple_types:
    for data_group in data_groups:
        files = file_dict[couple_type][data_group]
        # resampling
        n_resample_group = [10000]
        for n_resample in n_resample_group:
            # hyper-hyper parameters
            alpha_group = [4.0,8.0]
            for alpha in alpha_group:
                gamma_group = [4.0,8.0]
                for gamma in gamma_group:
                    # hyper-gaussian
                    mu = pgaussian_dict[couple_type][data_group][0]
                    lmda = pgaussian_dict[couple_type][data_group][1]
                    kappa = 0.25
                    nu = 4
                    # hyper-possion
                    k,theta = gamma_dict[couple_type][data_group]
                    
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
                            o.write(couple_type)
                            o.write("\n")
                            o.write(data_group)
                            o.write("\n")
                            o.write(files)
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
                            
            

