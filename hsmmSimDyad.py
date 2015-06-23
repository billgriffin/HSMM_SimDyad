#!/usr/bin/env python
import os
import wx
import numpy as np
from wx import xrc
from matplotlib import pyplot as plt

import pyhsmm
from pyhsmm import nicetext

# suppress  warnings
np.seterr(divide='ignore')

class HSMM():
    def __init__(self,couple_type):
        self.data = None
        self.couple_type = couple_type
        self.n_resampling = '1'
        self.alpha = '10.0'
        self.gamma = '10.0'
        self.mu = 'np.zeros(2)'
        self.lmbda = 'np.eye(2)'
        self.kappa = '0.1'
        self.nu = '4'
        '''gamma generates lamdba for poisson'''
        self.k = '5'     # MJohn uses 60; we use 5; for gamma - used for poisson hyper
        self.theta = '3' # gamma beta; theta here; rate parameter; hyper for poisson
        self.n_states = '20'
        self.n_trunc = '60'
        self.model = None
   
    def setData(self, data):
        self.data = data
        
    def equals(self, hsmm):
        self.n_resampling = hsmm.n_resampling     
        '''For transitions: hyperparameters follow the notation in Fox et al.
        gamma: concentration paramter for beta
        alpha: total mass concentration parameter for each row of trans matrix'''        
        self.alpha = hsmm.alpha              # alpha for transitions; DP(alpha,beta)
        self.gamma = hsmm.gamma        # generate beta for transitions; 
                                                               # beta ~ GEM(gamma); stick
        # xun: [0] OK, (alpha,gamma ) is for HDP of trans_distns, 
        # gamma is concentration  parameter for beta
        # see internal.transition.py   class HDPHSMMTransitions()
        #
        # the init_state_concentration is for DP with k components (states), and scalar
        # see basic.pybasicbayes.distributions.py  class Multinomial()
        
        self.mu = hsmm.mu                   #mu_0 in obs_hypparams
        self.lmbda = hsmm.lmbda        #sigma_0 in obs_hypparams
        self.kappa = hsmm.kappa        #kappa_0 in obs_hypparams
        self.nu = hsmm.nu                    #nu_0 in obs_hypparams
        
        self.k = hsmm.k                      #alpha_0 in gamma; prior for lambda; dur_hypparams
        self.theta = hsmm.theta        #beta_0 in gamma; prior for lambda; dur_hypparams
        
        self.n_states = hsmm.n_states
        self.n_trunc = hsmm.n_trunc
        
    def Run(self):
        Nmax = int(self.n_states)
        T = self.data[0].shape[0]
        n_resample = int(self.n_resampling)
        
        '''observation hyperparameters'''
        obs_hypparams = {'mu_0': eval(self.mu),
    # xun: [0.1] was lmbda_0                        
                        'sigma_0':eval(self.lmbda), 
                        'kappa_0':float(self.kappa),
                        'nu_0':float(self.nu)}
        
        '''duration hyperparameters; uses Gamma parameters (alpha, beta) to build Poisson(lambda)'''
    # xun: [0.2] was k,theta
        dur_hypparams = {'alpha_0': float(self.k),   # shape ;hsmm.k changed to alpha_0
                         'beta_0':float(self.theta)}           # rate (1/theta) ; hsmm.theta to beta_0
        
        obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(Nmax)] #xun: class changed
        dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in xrange(Nmax)]
        alpha = float(self.alpha); gamma = float(self.gamma);n_trunc=int(self.n_trunc)
        
        # xun: [0.3]  we deal with 4 distributions: 
        # obs_distns, Gaussian with a Gaussian conjugate prior  (conti)
        # dur_distns, Possion with a Gamma conjugate prior  (conti)
        # state_distns <--init_concentration, DP  (categorical, Multinomial with a DP conjugate prior) 
        # trans_distns <-- alpha,gamma,   HDP 
        model = pyhsmm.models.HSMM( # xun: class changed
            alpha=alpha,
            gamma=gamma,
            init_state_concentration=6.0, # xun: new add
            obs_distns=obs_distns,
            dur_distns=dur_distns,
            trunc=n_trunc) # alpha, gamma was 6.,6.; trunc = max duration
        
        for data in self.data:
            model.add_data(data)
            # print model.obs_distns[0].nu_0, model.obs_distns[0].lmbda_0, model.obs_distns[0].mu_0, model.obs_distns[0].kappa_0
            print model.obs_distns[0].nu_0, model.obs_distns[0].sigma_0, model.obs_distns[0].mu_0, model.obs_distns[0].kappa_0
            
        for i in xrange(n_resample):
            model.resample_model() # xun: function change
            
        # need this #model.print_results(s, x, used_states, dur_distns, trans_distn)
        # get state distn information
        # expect_dur =    []  # not needed; lambda used from poisson
        grp_means_D1 =  []
        grp_means_D2 =  []
        grp_var_D1   =  []
        grp_var_D2   =  []
        grp_covar =     []
        state_idx =     []
        lmbda_values =  [] 
        
        for i in range(len(model.obs_distns)):  #see models compare
            state_idx.append(i)
            lmbda_values.append(dur_distns[i].lmbda)
            if model.obs_distns[i].mu[0] < 0.:
                model.obs_distns[i].mu[0] = 0.
            grp_means_D1.append(np.round(model.obs_distns[i].mu[0]))
            if model.obs_distns[i].mu[1] < 0.:
                model.obs_distns[i].mu[1] = 0.
            grp_means_D2.append(np.round(model.obs_distns[i].mu[1]))
            grp_var_D1.append(model.obs_distns[i].sigma[0][0])
            grp_var_D2.append(model.obs_distns[i].sigma[1][1])
            grp_covar.append(model.obs_distns[i].sigma[0][1])
        state_values = zip(grp_means_D1, grp_means_D2)
        sys_dict =  dict(zip(state_idx, state_values))
            
        self.T = T
        self.model = model
        self.states_list = model.states_list
        self.sys_dict = sys_dict
        self.grp_info = zip(state_idx,lmbda_values,grp_means_D1,grp_means_D2,grp_var_D1,grp_var_D2, grp_covar)
        self.states_dict_list,self.results_list = self.model.prerun()
        
        wx.MessageBox('build HSMM model done.','Info',wx.OK|wx.ICON_INFORMATION)
        
    def GetSystemStates(self):
        content = '\n*********************************************\n'
        content += 'System states:'
        content += '\n*********************************************\n'
        headers =  'State', 'Duration:', ' D1:', '   D2:', '   D1:Var', '   D2:Var', '     Cov'
        content += '\n'
        content += nicetext.SimpleTable(
            self.grp_info,
            headers,
            title = 'Group Level: States, Expected Duration, Means, Variance, Covariance' ,
            fmt={'data_fmt':['%g','%1.3f', '%1.1f', '%1.1f', '%1.3f', '%1.3f', '%1.3f']}).as_text()
        
        """
        results, labels =  self.model.generate(self.T, keep=False)
        content += '\nStates Module: Generated Sequence; General Representation\n'
        labelout =  labels.tolist()
        for i in labelout:
            content += '%s\n'% labelout[i]
        """
        
        content += '\n*********************************************\n'
        content += '%s Individual states:' % len(self.results_list)
        content += '\n*********************************************\n'
        for results in self.results_list:
            content += results
        return content
        
    def PlotResults(self, selections):
        title="HSMM model for %s couple (%s files)" %(self.couple_type,len(self.data))
        plt.figure(figsize=(12,12))
        plt.title(title)
        self.model.selected_plot(selections,title)
        plt.show()
        
    def simulate_individual(self,index):
        s = self.states_list[index]
        subject_dict = self.states_dict_list[index]
        #s.generate()
        sub_state_seq = []
        sub_state_seq_X =  []
        sub_state_seq_Y =  []
        for i in range(len(s.stateseq)):
            sub_state_seq.append(subject_dict[s.stateseq[i]])
        for i in range(len(sub_state_seq)):    
            sub_state_seq_X.append(sub_state_seq[i][0])
            sub_state_seq_Y.append(sub_state_seq[i][1])   
        plt.figure(figsize=(15, 10))
        plt.plot(sub_state_seq_X, label ='D1', color='b' )
        plt.plot(sub_state_seq_Y, label = 'D2', color='r')
        plt.title('Plausible Sequence of States for individual subject %s in Group: %s'%(index,self.couple_type))
        plt.legend()
        plt.xlim((0, self.T))
        plt.ylim((0, 9))
        plt.show()
        
    def simulate_generic(self):
        T = self.T
        sys_dict = self.sys_dict
        results, labels =  self.model.generate(T, keep=False)
        state_seq = []
        state_seq_X =  []
        state_seq_Y =  []
        for i in range(T):
            state_seq.append(sys_dict[labels[i]])
        for i in range(len(state_seq)):    
            state_seq_X.append(state_seq[i][0])
            state_seq_Y.append(state_seq[i][1])
        plt.figure(figsize=(15, 10))
        plt.plot(state_seq_X, label ='D1', color='b' )
        plt.plot(state_seq_Y, label = 'D2', color='r')
        plt.title('Plausible Sequence of States for Satisfaction Group: %s'%self.couple_type)
        plt.legend()
        plt.xlim((0, self.T))
        plt.ylim((0, 9))
        plt.show()
        
    def Simulation(self,event):
        button = event.GetEventObject()
        

class HSMMDialog():
    def __init__(self, data, hsmm, dialog, hsmm_list):
        self.hsmm = hsmm
        self.hsmm.setData(data)
        self.dialog = dialog
        self.hsmm_list =hsmm_list
        self.btnHSMM = xrc.XRCCTRL(self.dialog, 'btnRunHSMM')
        self.btnHSMM.Bind(wx.EVT_BUTTON, self.runHSMM)
        self.btnApplyConfToAll = xrc.XRCCTRL(self.dialog, 'btnApplyConfToAll')
        self.btnApplyConfToAll.Bind(wx.EVT_BUTTON, self.applyConfigurationToAll)
        self.txt_n_resampling = xrc.XRCCTRL(self.dialog, 'n_sampling')
        self.txt_alpha = xrc.XRCCTRL(self.dialog, 'alpha')
        self.txt_gamma = xrc.XRCCTRL(self.dialog, 'gamma')
        self.txt_mu = xrc.XRCCTRL(self.dialog, 'mu')
        self.txt_lambda = xrc.XRCCTRL(self.dialog, 'lambda')
        self.txt_kappa = xrc.XRCCTRL(self.dialog, 'kappa')
        self.txt_nu = xrc.XRCCTRL(self.dialog, 'nu')
        self.txt_k = xrc.XRCCTRL(self.dialog, 'k')
        self.txt_theta = xrc.XRCCTRL(self.dialog, 'theta')
        self.txt_n_states = xrc.XRCCTRL(self.dialog, 'n_states')
        self.txt_n_trunc = xrc.XRCCTRL(self.dialog, 'n_trunc')
        self.lbl_info = xrc.XRCCTRL(self.dialog, 'txtConfTitle')
        self.lbl_info.SetLabel("%s files used in HSMM model for %s couple" %(len(self.hsmm.data),self.hsmm.couple_type))
        
        self.txt_n_states.SetValue(self.hsmm.n_states)
        self.txt_n_resampling.SetValue(self.hsmm.n_resampling)
        self.txt_alpha.SetValue(self.hsmm.alpha)
        self.txt_gamma.SetValue(self.hsmm.gamma)
        self.txt_mu.SetValue(self.hsmm.mu)
        self.txt_lambda.SetValue(self.hsmm.lmbda)
        self.txt_kappa.SetValue(self.hsmm.kappa)
        self.txt_nu.SetValue(self.hsmm.nu)
        self.txt_k.SetValue(self.hsmm.k)
        self.txt_theta.SetValue(self.hsmm.theta)
        self.txt_n_trunc.SetValue(self.hsmm.n_trunc)
        
    def Show(self):
        self.dialog.Fit()
        return self.dialog.Show()
    
    def Destroy(self):
        self.dialog.Destroy()
       
    def applyConfigurationToAll(self, event):
        if self.hsmm_list:
            for tmp_hsmm in self.hsmm_list:
                if tmp_hsmm != self.hsmm:
                    tmp_hsmm.equals(self.hsmm)
    
    def runHSMM(self, event):
        self.hsmm.n_states = self.txt_n_states.GetValue()
        self.hsmm.n_resampling = self.txt_n_resampling.GetValue()
        self.hsmm.alpha = self.txt_alpha.GetValue()
        self.hsmm.gamma = self.txt_gamma.GetValue()
        self.hsmm.mu = self.txt_mu.GetValue()
        self.hsmm.lmbda = self.txt_lambda.GetValue()
        self.hsmm.kappa = self.txt_kappa.GetValue()
        self.hsmm.nu = self.txt_nu.GetValue()
        self.hsmm.k = self.txt_k.GetValue()
        self.hsmm.theta = self.txt_theta.GetValue()
        self.hsmm.n_trunc = self.txt_n_trunc.GetValue()
        self.hsmm.Run()
        
    
        
class HSMMSimDyadApp(wx.App):
    def OnInit(self):
        self.res = xrc.XmlResource("hsmmSimDyad.xrc")
 
        frame = self.res.LoadFrame(None, 'HSMMSimDyadFrame')
        self.frame = frame
        """
        self.hsmm_dialog = self.res.LoadDialog(None, 'HSMMConfigure')
        self.btnExitConfig = xrc.XRCCTRL(self.hsmm_dialog, 'btnExit')
        self.hsmm_dialog.Bind(wx.EVT_BUTTON, self.Exit)
        """
        self.hsmmHH=HSMM("High");self.hsmmMM=HSMM("Median");self.hsmmLL=HSMM("Low")
        self.hsmmCluster1HH=HSMM("High Cluster1");self.hsmmCluster2HH=HSMM("High Cluster2")
        self.hsmmCluster1MM=HSMM("Median Cluster1");self.hsmmCluster2MM=HSMM("Median Cluster2")
        self.hsmmCluster1LL=HSMM("Low Cluster1");self.hsmmCluster2LL=HSMM("Low Cluster2")
        self.hsmm_list = [self.hsmmHH,self.hsmmMM,self.hsmmLL,self.hsmmCluster1HH,self.hsmmCluster1MM,
                          self.hsmmCluster1LL,self.hsmmCluster2HH,self.hsmmCluster2MM,self.hsmmCluster2LL]
        
        load_raw_files_HH = xrc.XRCCTRL(frame, 'loadRawFilesHH')
        load_raw_files_MM = xrc.XRCCTRL(frame, 'loadRawFilesMM')
        load_raw_files_LL = xrc.XRCCTRL(frame, 'loadRawFilesLL')
        load_raw_files_HH.Bind(wx.EVT_BUTTON, self.load_raw_files_HH)
        load_raw_files_MM.Bind(wx.EVT_BUTTON, self.load_raw_files_MM)
        load_raw_files_LL.Bind(wx.EVT_BUTTON, self.load_raw_files_LL)
          
        self.btnRunClusteringHH = xrc.XRCCTRL(frame, 'runClusteringHH')
        self.btnRunClusteringMM = xrc.XRCCTRL(frame, 'runClusteringMM')
        self.btnRunClusteringLL = xrc.XRCCTRL(frame, 'runClusteringLL')
        self.btnRunClusteringHH.Bind(wx.EVT_BUTTON, self.runClusteringHH)
        self.btnRunClusteringMM.Bind(wx.EVT_BUTTON, self.runClusteringMM)
        self.btnRunClusteringLL.Bind(wx.EVT_BUTTON, self.runClusteringLL)     
        
        self.btnHsmmModelHH = xrc.XRCCTRL(frame, 'hsmmModelHH')
        self.btnHsmmModelMM = xrc.XRCCTRL(frame, 'hsmmModelMM')
        self.btnHsmmModelLL = xrc.XRCCTRL(frame, 'hsmmModelLL')
        self.btnHsmmModelHH.Bind(wx.EVT_BUTTON, self.runHSMMModelHH)
        self.btnHsmmModelMM.Bind(wx.EVT_BUTTON, self.runHSMMModelMM)
        self.btnHsmmModelLL.Bind(wx.EVT_BUTTON, self.runHSMMModelLL)
        
        self.btnPlotResultsHH = xrc.XRCCTRL(frame, 'plotHH')
        self.btnPlotResultsMM = xrc.XRCCTRL(frame, 'plotMM')
        self.btnPlotResultsLL = xrc.XRCCTRL(frame, 'plotLL')
        self.btnPlotResultsHH.Bind(wx.EVT_BUTTON, self.plotResultsHH)
        self.btnPlotResultsMM.Bind(wx.EVT_BUTTON, self.plotResultsMM)
        self.btnPlotResultsLL.Bind(wx.EVT_BUTTON, self.plotResultsLL)
        
        self.btnShowResultsHH = xrc.XRCCTRL(frame, 'showResultsHH')
        self.btnShowResultsMM = xrc.XRCCTRL(frame, 'showResultsMM')
        self.btnShowResultsLL = xrc.XRCCTRL(frame, 'showResultsLL')
        self.btnShowResultsHH.Bind(wx.EVT_BUTTON, self.showResultsHH)
        self.btnShowResultsMM.Bind(wx.EVT_BUTTON, self.showResultsMM)
        self.btnShowResultsLL.Bind(wx.EVT_BUTTON, self.showResultsLL)
        
        self.btnRunSimHH = xrc.XRCCTRL(frame, 'runSimHH')
        self.btnRunSimMM = xrc.XRCCTRL(frame, 'runSimMM')
        self.btnRunSimLL = xrc.XRCCTRL(frame, 'runSimLL')
        self.btnRunSimHH.Bind(wx.EVT_BUTTON, self.runSimHH)
        self.btnRunSimMM.Bind(wx.EVT_BUTTON, self.runSimMM)
        self.btnRunSimLL.Bind(wx.EVT_BUTTON, self.runSimLL)
        
        self.btnHsmmModelCluster1HH = xrc.XRCCTRL(frame, 'hsmmModelCluster1HH')
        self.btnHsmmModelCluster1MM = xrc.XRCCTRL(frame, 'hsmmModelCluster1MM')
        self.btnHsmmModelCluster1LL = xrc.XRCCTRL(frame, 'hsmmModelCluster1LL')
        self.btnHsmmModelCluster1HH.Bind(wx.EVT_BUTTON, self.runHSMMModelCluster1HH)
        self.btnHsmmModelCluster1MM.Bind(wx.EVT_BUTTON, self.runHSMMModelCluster1MM)
        self.btnHsmmModelCluster1LL.Bind(wx.EVT_BUTTON, self.runHSMMModelCluster1LL)
        
        self.btnPlotResultsCluster1HH = xrc.XRCCTRL(frame, 'plotCluster1HH')
        self.btnPlotResultsCluster1MM = xrc.XRCCTRL(frame, 'plotCluster1MM')
        self.btnPlotResultsCluster1LL = xrc.XRCCTRL(frame, 'plotCluster1LL')
        self.btnPlotResultsCluster1HH.Bind(wx.EVT_BUTTON, self.plotResultsCluster1HH)
        self.btnPlotResultsCluster1MM.Bind(wx.EVT_BUTTON, self.plotResultsCluster1MM)
        self.btnPlotResultsCluster1LL.Bind(wx.EVT_BUTTON, self.plotResultsCluster1LL)
        
        self.btnShowResultsCluster1HH = xrc.XRCCTRL(frame, 'showResultsCluster1HH')
        self.btnShowResultsCluster1MM = xrc.XRCCTRL(frame, 'showResultsCluster1MM')
        self.btnShowResultsCluster1LL = xrc.XRCCTRL(frame, 'showResultsCluster1LL')
        self.btnShowResultsCluster1HH.Bind(wx.EVT_BUTTON, self.showResultsCluster1HH)
        self.btnShowResultsCluster1MM.Bind(wx.EVT_BUTTON, self.showResultsCluster1MM)
        self.btnShowResultsCluster1LL.Bind(wx.EVT_BUTTON, self.showResultsCluster1LL)
        
        self.btnRunSimCluster1HH = xrc.XRCCTRL(frame, 'runSimCluster1HH')
        self.btnRunSimCluster1MM = xrc.XRCCTRL(frame, 'runSimCluster1MM')
        self.btnRunSimCluster1LL = xrc.XRCCTRL(frame, 'runSimCluster1LL')
        self.btnRunSimCluster1HH.Bind(wx.EVT_BUTTON, self.runSimCluster1HH)
        self.btnRunSimCluster1MM.Bind(wx.EVT_BUTTON, self.runSimCluster1MM)
        self.btnRunSimCluster1LL.Bind(wx.EVT_BUTTON, self.runSimCluster1LL)
        
        self.btnHsmmModelCluster2HH = xrc.XRCCTRL(frame, 'hsmmModelCluster2HH')
        self.btnHsmmModelCluster2MM = xrc.XRCCTRL(frame, 'hsmmModelCluster2MM')
        self.btnHsmmModelCluster2LL = xrc.XRCCTRL(frame, 'hsmmModelCluster2LL')
        self.btnHsmmModelCluster2HH.Bind(wx.EVT_BUTTON, self.runHSMMModelCluster2HH)
        self.btnHsmmModelCluster2MM.Bind(wx.EVT_BUTTON, self.runHSMMModelCluster2MM)
        self.btnHsmmModelCluster2LL.Bind(wx.EVT_BUTTON, self.runHSMMModelCluster2LL)
        
        self.btnPlotResultsCluster2HH = xrc.XRCCTRL(frame, 'plotCluster2HH')
        self.btnPlotResultsCluster2MM = xrc.XRCCTRL(frame, 'plotCluster2MM')
        self.btnPlotResultsCluster2LL = xrc.XRCCTRL(frame, 'plotCluster2LL')
        self.btnPlotResultsCluster2HH.Bind(wx.EVT_BUTTON, self.plotResultsCluster2HH)
        self.btnPlotResultsCluster2MM.Bind(wx.EVT_BUTTON, self.plotResultsCluster2MM)
        self.btnPlotResultsCluster2LL.Bind(wx.EVT_BUTTON, self.plotResultsCluster2LL)
        
        self.btnShowResultsCluster2HH = xrc.XRCCTRL(frame, 'showResultsCluster2HH')
        self.btnShowResultsCluster2MM = xrc.XRCCTRL(frame, 'showResultsCluster2MM')
        self.btnShowResultsCluster2LL = xrc.XRCCTRL(frame, 'showResultsCluster2LL')
        self.btnShowResultsCluster2HH.Bind(wx.EVT_BUTTON, self.showResultsCluster2HH)
        self.btnShowResultsCluster2MM.Bind(wx.EVT_BUTTON, self.showResultsCluster2MM)
        self.btnShowResultsCluster2LL.Bind(wx.EVT_BUTTON, self.showResultsCluster2LL)
        
        self.btnRunSimCluster2HH = xrc.XRCCTRL(frame, 'runSimCluster2HH')
        self.btnRunSimCluster2MM = xrc.XRCCTRL(frame, 'runSimCluster2MM')
        self.btnRunSimCluster2LL = xrc.XRCCTRL(frame, 'runSimCluster2LL')
        self.btnRunSimCluster2HH.Bind(wx.EVT_BUTTON, self.runSimCluster2HH)
        self.btnRunSimCluster2MM.Bind(wx.EVT_BUTTON, self.runSimCluster2MM)
        self.btnRunSimCluster2LL.Bind(wx.EVT_BUTTON, self.runSimCluster2LL)
       
        self.clbFilesHH = xrc.XRCCTRL(frame, 'clbFilesHH')
        self.clbFilesMM = xrc.XRCCTRL(frame, 'clbFilesMM')
        self.clbFilesLL = xrc.XRCCTRL(frame, 'clbFilesLL')
        self.clbFilesHH.Bind(wx.EVT_CHECKLISTBOX, self.checkFilesHH)
        self.clbFilesMM.Bind(wx.EVT_CHECKLISTBOX, self.checkFilesMM)
        self.clbFilesLL.Bind(wx.EVT_CHECKLISTBOX, self.checkFilesLL)
        
        self.clbFilesCluster1HH = xrc.XRCCTRL(frame, 'clbCluster1HH')
        self.clbFilesCluster1MM = xrc.XRCCTRL(frame, 'clbCluster1MM')
        self.clbFilesCluster1LL = xrc.XRCCTRL(frame, 'clbCluster1LL')
        self.clbFilesCluster1HH.Bind(wx.EVT_CHECKLISTBOX, self.checkFilesCluster1HH)
        self.clbFilesCluster1MM.Bind(wx.EVT_CHECKLISTBOX, self.checkFilesCluster1MM)
        self.clbFilesCluster1LL.Bind(wx.EVT_CHECKLISTBOX, self.checkFilesCluster1LL)
        
        self.clbFilesCluster2HH = xrc.XRCCTRL(frame, 'clbCluster2HH')
        self.clbFilesCluster2MM = xrc.XRCCTRL(frame, 'clbCluster2MM')
        self.clbFilesCluster2LL = xrc.XRCCTRL(frame, 'clbCluster2LL')
        self.clbFilesCluster2HH.Bind(wx.EVT_CHECKLISTBOX, self.checkFilesCluster2HH)
        self.clbFilesCluster2MM.Bind(wx.EVT_CHECKLISTBOX, self.checkFilesCluster2MM)
        self.clbFilesCluster2LL.Bind(wx.EVT_CHECKLISTBOX, self.checkFilesCluster2LL)
        
        frame.Show()
        return True

    def Exit(self,event):
        event.EventObject.Parent.Parent.Destroy()
    
    #------------------------------------------------------
    def toggleControls(self,couple_type):
        eval('self.btnHsmmModelCluster1%s.Enable()'%couple_type)
        eval('self.btnHsmmModelCluster2%s.Enable()'%couple_type)
        eval('self.btnShowResultsCluster1%s.Enable()'%couple_type)
        eval('self.btnShowResultsCluster2%s.Enable()'%couple_type)
        eval('self.btnRunSimCluster1%s.Enable()'%couple_type)
        eval('self.btnRunSimCluster2%s.Enable()'%couple_type)
        eval('self.btnPlotResultsCluster2%s.Enable()'%couple_type)
        eval('self.btnPlotResultsCluster1%s.Enable()'%couple_type)
        
    def select_read_files(self,checklistbox):
        wildcard = "Text files (*.txt)|*.txt|"     \
           "CSV files(*.csv)|*.csv|" \
           "Data files (*.dat)|*.dat|"    \
           "All files (*.*)|*.*"
        paths = []
        dlg = wx.FileDialog(
            None, message="Choose raw data files",
            defaultDir=os.getcwd(), 
            defaultFile="",
            wildcard=wildcard,
            style=wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR)
        if dlg.ShowModal() == wx.ID_OK:
            paths = dlg.GetPaths()
        dlg.Destroy()
        dataset = {}
        for f in paths:
            data = np.transpose(np.loadtxt(f, usecols=(0,1),unpack=True))
            dataset[f] = data
        checklistbox.Set(paths)
        for i in range(len(paths)):
            checklistbox.Check(i)
        return paths,dataset
    
    def toggleMainButons(self,couple_type):
        eval('self.btnRunClustering%s.Enable()'%couple_type)
        eval('self.btnHsmmModel%s.Enable()'%couple_type)
        eval('self.btnShowResults%s.Enable()'%couple_type)
        eval('self.btnRunSim%s.Enable()'%couple_type)
        eval('self.btnPlotResults%s.Enable()'%couple_type)
        
    def load_raw_files_HH(self,event):
        self.HH_files,self.HH_dataset = self.select_read_files(self.clbFilesHH)
        self.toggleMainButons('HH')
       
    def load_raw_files_MM(self,event):
        self.MM_files,self.MM_dataset = self.select_read_files(self.clbFilesMM)
        self.toggleMainButons('MM')
    
    def load_raw_files_LL(self,event):
        self.LL_files,self.LL_dataset = self.select_read_files(self.clbFilesLL)
        self.toggleMainButons('LL')
    
    def get_selected_dataset(self, dataset, checklistbox):
        selected_paths = []
        selected_data = []
        n = checklistbox.GetCount()
        for i in range(n):
            if checklistbox.GetSelection():
                path = checklistbox.GetString(i)
                data = dataset[path]
                selected_data.append(data)
                selected_paths.append(path)
                
        """todo"""
        self.hsmm_dialog = self.res.LoadDialog(None, 'HSMMConfigure')
        self.btnExitConfig = xrc.XRCCTRL(self.hsmm_dialog, 'btnExit')
        self.hsmm_dialog.Bind(wx.EVT_BUTTON, self.Exit)
        return selected_paths,selected_data
                           
    #------------------------------------------------------
    def clustering(self, dataset):
        import pyeeg as eg
        from pyhsmm.util.plot import pca_project_data
        def extract_features(couple_data):
            pca1 =  pca_project_data(couple_data, 1) #take 1st pca dimension
            pca1_mean =  np.mean(pca1, axis=0) #
            pca1_std   = np.std(pca1, axis=0)  # 
            pca1_med = np.median(pca1,  axis=0) #
            features = []
            def sinuosity_deviation_features(seq, mean, std):
                sinuosity_dict = {"A":0, "B":0, "C":0}
                deviation_dict = {"I":0, "II":0, "III":0}
                sinuosity_deviation_dict = {"A-I":0,"A-II":0,"A-III":0,"B-I":0,"B-II":0,"B-III":0,"C-I":0,"C-II":0,"C-III":0}
                n = len(seq)
                for i in range(1,n-1):
                    current_af = seq[i]
                    prev_af = seq[i-1]
                    next_af = seq[i+1]
                    sinu = abs((next_af - current_af) + (current_af - prev_af))
                    if sinu == 0:      label1 = "A"
                    elif 0< sinu <= 1: label1 = "B"
                    else:              label1 = 'C'
                    sinuosity_dict[label1] += 1
                    devi = abs(current_af - mean)
                    close =  std / 2
                    if devi <= close: label2 = "I"
                    elif devi <= std: label2 = "II" 
                    elif devi > std:  label2 = "III" 
                    deviation_dict[label2] += 1
                    sinuosity_deviation_dict["%s-%s"%(label1,label2)] += 1
                return sinuosity_deviation_dict.values()    
            n = len(pca1)
            pca1_sinuosity_deviation = sinuosity_deviation_features( pca1, pca1_mean,  pca1_std )
            features += list(np.array(pca1_sinuosity_deviation)/float(n-2))
            seq =  pca1
            dfa = eg.dfa(seq); pfd = eg.pfd(seq)
            apen = eg.ap_entropy(seq,1,np.std(seq)*.2)
            svden = eg.svd_entropy(seq, 2, 2)
            features += [pca1_mean, pca1_med, pca1_std, dfa, pfd, apen, svden]
            return features
        def run_clustering(X,k):
            from sklearn import cluster, datasets
            from sklearn.neighbors import kneighbors_graph
            from sklearn.preprocessing import Scaler
            from sklearn.decomposition import PCA
            X = Scaler().fit_transform(X)
            pca = PCA(n_components=6) 
            pca.fit(X) # first 6 reach 93.6%, using pca.explained_variance_ratio_
            #print sum(pca.explained_variance_ratio_)
            X = pca.fit_transform(X)
            connectivity = kneighbors_graph(X, n_neighbors=8)
            connectivity = 0.5 * (connectivity + connectivity.T)
            algorithm = cluster.Ward(n_clusters=k, connectivity=connectivity)
            algorithm.fit(X)
            y_pred = algorithm.labels_.astype(np.int)
            return y_pred
        X = []
        for data in dataset:
            features = extract_features(data)
            X.append(features)
        X = np.array(X);k=2
        return run_clustering(X,k)
    
    def get_file_group(self, labels, paths):
        unique_labels = set(labels)
        n_labels = len(unique_labels)
        file_group = dict([(lbl,[]) for lbl in unique_labels])
        for i,lbl in enumerate(labels):
            path = paths[i]
            file_group[lbl].append(path)
        return file_group.values()
            
    def runClustering(self, dataset, checkboxlist, checkboxlist1, checkboxlist2):
        if len(dataset) < 3:
            wx.MessageBox('Please select more than 3 files for clusteirng.','Info',wx.OK|wx.ICON_INFORMATION)
            return
        selected_paths,selected_data = self.get_selected_dataset(dataset, checkboxlist)
        labels = self.clustering(selected_data)
        #labels = [0,0,0,0,1,0,0,0,1,1]
        file_group = self.get_file_group(labels, selected_paths)
        checkboxlist1.Set(file_group[0]) 
        checkboxlist2.Set(file_group[1]) 
        for cbl in [checkboxlist1,checkboxlist2]:
            n = cbl.GetCount()
            for i in range(n):
                cbl.Check(i)
        checkboxlist1.Enable(True)
        checkboxlist2.Enable(True)
        
    def runClusteringHH(self,event):
        self.runClustering(self.HH_dataset, self.clbFilesHH, self.clbFilesCluster1HH, self.clbFilesCluster2HH)
        self.toggleControls("HH")
        
    def runClusteringMM(self,event):
        self.runClustering(self.MM_dataset, self.clbFilesMM, self.clbFilesCluster1MM, self.clbFilesCluster2MM)
        self.toggleControls("MM")
    
    def runClusteringLL(self,event):
        self.runClustering(self.LL_dataset, self.clbFilesLL, self.clbFilesCluster1LL, self.clbFilesCluster2LL)
        self.toggleControls("LL")

    #------------------------------------------------------
    def runHSMMModelHH(self, event):
        selected_paths,selected_data = self.get_selected_dataset(self.HH_dataset, self.clbFilesHH)
        hsmmDlg = HSMMDialog(selected_data, self.hsmmHH, self.hsmm_dialog, self.hsmm_list)
        hsmmDlg.Show()
    
    def runHSMMModelMM(self, event):
        selected_paths,selected_data = self.get_selected_dataset(self.MM_dataset, self.clbFilesMM)
        hsmmDlg = HSMMDialog(selected_data, self.hsmmMM, self.hsmm_dialog, self.hsmm_list)
        hsmmDlg.Show()
    
    def runHSMMModelLL(self, event):
        selected_paths,selected_data = self.get_selected_dataset(self.LL_dataset, self.clbFilesLL)
        hsmmDlg = HSMMDialog(selected_data, self.hsmmLL, self.hsmm_dialog, self.hsmm_list)
        hsmmDlg.Show()
    
    def runHSMMModelCluster1HH(self, event):
        selected_paths,selected_data = self.get_selected_dataset(self.HH_dataset, self.clbFilesCluster1HH)
        hsmmDlg = HSMMDialog(selected_data, self.hsmmCluster1HH, self.hsmm_dialog, self.hsmm_list)
        hsmmDlg.Show()
    
    def runHSMMModelCluster1MM(self, event):
        selected_paths,selected_data = self.get_selected_dataset(self.MM_dataset, self.clbFilesCluster1MM)
        hsmmDlg = HSMMDialog(selected_data, self.hsmmCluster1MM, self.hsmm_dialog, self.hsmm_list)
        hsmmDlg.Show()
    
    def runHSMMModelCluster1LL(self, event):
        selected_paths,selected_data = self.get_selected_dataset(self.LL_dataset, self.clbFilesCluster1LL)
        hsmmDlg = HSMMDialog(selected_data, self.hsmmCluster1LL, self.hsmm_dialog, self.hsmm_list)
        hsmmDlg.Show()
    
    def runHSMMModelCluster2HH(self, event):
        selected_paths,selected_data = self.get_selected_dataset(self.HH_dataset, self.clbFilesCluster2HH)
        hsmmDlg = HSMMDialog(selected_data, self.hsmmCluster2HH, self.hsmm_dialog, self.hsmm_list)
        hsmmDlg.Show()
    
    def runHSMMModelCluster2MM(self, event):
        selected_paths,selected_data = self.get_selected_dataset(self.HH_dataset, self.clbFilesCluster2MM)
        hsmmDlg = HSMMDialog(selected_data, self.hsmmCluster2MM, self.hsmm_dialog, self.hsmm_list)
        hsmmDlg.Show()
        pass
    
    def runHSMMModelCluster2LL(self, event):
        selected_paths,selected_data = self.get_selected_dataset(self.LL_dataset, self.clbFilesCluster2LL)
        hsmmDlg = HSMMDialog(selected_data, self.hsmmCluster2LL, self.hsmm_dialog, self.hsmm_list)
        hsmmDlg.Show()
    
    #------------------------------------------------------
    def saveResults(self, event):
        result_dlg = event.EventObject.Parent
        txtResult = xrc.XRCCTRL(result_dlg, 'txtResult')
        content = txtResult.GetValue()
        dlg = wx.FileDialog(
            None, message="Save results in a file...", defaultDir=os.getcwd(), 
            defaultFile='results.txt', 
            wildcard="text file (*.txt)|*.txt|csv file (*.csv)|*.csv|All files (*.*)|*.*", 
            style=wx.SAVE)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            o = open(path,'w')
            o.write(content)
            o.close()
        dlg.Destroy()
    
    def showResults(self,hsmm_model):
        if hsmm_model.model== None:
            return
        result_dlg = self.res.LoadDialog(None, 'HSMMResultDlg')
        result_dlg_toolbar = xrc.XRCCTRL(result_dlg, 'toolbar')
        result_dlg_toolbar.Bind(wx.EVT_TOOL, self.saveResults,id=xrc.XRCID('toolSave'))
        
        txtResult = xrc.XRCCTRL(result_dlg, 'txtResult')
        txtResult.SetValue(hsmm_model.GetSystemStates())
        result_dlg.Fit()
        result_dlg.Show()
        
    def showResultsHH(self, event):
        self.showResults(self.hsmmHH)
    
    def showResultsMM(self, event):
        self.showResults(self.hsmmMM)
    
    def showResultsLL(self, event):
        self.showResults(self.hsmmLL)
   
    def showResultsCluster1HH(self, event):
        self.showResults(self.hsmmCluster1HH)
    
    def showResultsCluster1MM(self, event):
        self.showResults(self.hsmmCluster1MM)
    
    def showResultsCluster1LL(self, event):
        self.showResults(self.hsmmCluster1LL)
    
    def showResultsCluster2HH(self, event):
        self.showResults( self.hsmmCluster2HH)
    
    def showResultsCluster2MM(self, event):
        self.showResults(self.hsmmCluster2MM)
    
    def showResultsCluster2LL(self, event):
        self.showResults(self.hsmmCluster2LL)
    
    #------------------------------------------------------
    def plotResults(self, model):
        if model.model == None:
            return
        selections = [0,1,2,3]
        lst = ["Observation Distrubtions","State Sequences","Durations","State Traces","Original Data"]
        dlg = wx.MultiChoiceDialog(None, "Select plots to display", "Plots", lst)
        dlg.SetSelections(selections)
        if dlg.ShowModal() == wx.ID_OK:
            selections = dlg.GetSelections()
        dlg.Destroy()
        
        model.PlotResults(selections)
        
    def plotResultsHH(self, event):
        self.plotResults(self.hsmmHH)
    
    def plotResultsMM(self, event):
        self.plotResults(self.hsmmMM)
    
    def plotResultsLL(self, event):
        self.plotResults(self.hsmmMM)
    
    def plotResultsCluster1HH(self, event):
        self.plotResults(self.hsmmCluster1HH)
    
    def plotResultsCluster1MM(self, event):
        self.plotResults(self.hsmmCluster1MM)
    
    def plotResultsCluster1LL(self, event):
        self.plotResults(self.hsmmCluster1LL)
    
    def plotResultsCluster2HH(self, event):
        self.plotResults(self.hsmmCluster2HH)
    
    def plotResultsCluster2MM(self, event):
        self.plotResults(self.hsmmCluster2MM)
    
    def plotResultsCluster2LL(self, event):
        self.plotResults(self.hsmmCluster2LL)
        
    #------------------------------------------------------
    def runSimulation(self, event, model,runfunction):
        if model.model == None:
            return
        menu = wx.Menu()
        start_ID = 10000
        n = len(model.data)
        for i in range(n):
            menu.Append(start_ID,'couple index ' +str(i))
            start_ID += 1
        menu.Append(start_ID, "Generic")
        start_ID = 10000
        for i in range(n+1):
            self.Bind(wx.EVT_MENU, runfunction, id=start_ID)
            start_ID += 1
        button = event.GetEventObject()
        pos = button.Position 
        self.frame.PopupMenu(menu, pos)
                      
    def runSimHH(self, event):
        self.runSimulation(event,self.hsmmHH,self.run_simulationHH)
        
    def runSimMM(self, event):
        self.runSimulation(event,self.hsmmMM,self.run_simulationMM)
    
    def runSimLL(self, event):
        self.runSimulation(event,self.hsmmLL,self.run_simulationLL)
    
    def runSimCluster1HH(self, event):
        self.runSimulation(event,self.hsmmCluster1HH,self.run_simulationCluster1HH)
    
    def runSimCluster1MM(self, event):
        self.runSimulation(event,self.hsmmCluster1MM,self.run_simulationCluster1MM)
    
    def runSimCluster1LL(self, event):
        self.runSimulation(event,self.hsmmCluster1LL,self.run_simulationCluster1LL)
    
    def runSimCluster2HH(self, event):
        self.runSimulation(event,self.hsmmCluster2HH,self.run_simulationCluster2HH)
    
    def runSimCluster2MM(self, event):
        self.runSimulation(event,self.hsmmCluster2MM,self.run_simulationCluster2MM)
    
    def runSimCluster2LL(self, event):
        self.runSimulation(event,self.hsmmCluster2LL,self.run_simulationCluster2LL)
    
    #------------------------------------------------------
    def checkFilesHH(self, event):
        pass
      
    def checkFilesMM(self, event):
        pass
    
    def checkFilesLL(self, event):
        pass
    
    def checkFilesCluster1HH(self, event):
        pass
      
    def checkFilesCluster1MM(self, event):
        pass
    
    def checkFilesCluster1LL(self, event):
        pass
    
    def checkFilesCluster2HH(self, event):
        pass
      
    def checkFilesCluster2MM(self, event):
        pass
    
    def checkFilesCluster2LL(self, event):
        pass
    
    def run_simulationHH(self, event):
        model = self.hsmmHH
        index = event.Id - 10000
        if index < len(model.data):model.simulate_individual(index)
        else:model.simulate_generic()
        
    def run_simulationMM(self, event):
        model = self.hsmmMM
        index = event.Id - 10000
        if index < len(model.data):model.simulate_individual(index)
        else:model.simulate_generic()
        
    def run_simulationLL(self, event):
        model = self.hsmmLL
        index = event.Id - 10000
        if index < len(model.data):model.simulate_individual(index)
        else:model.simulate_generic()
        
    def run_simulationCluster1HH(self, event):
        model = self.hsmmCluster1HH
        index = event.Id - 10000
        if index < len(model.data):model.simulate_individual(index)
        else:model.simulate_generic()
        
    def run_simulationCluster1MM(self, event):
        model = self.hsmmCluster1MM
        index = event.Id - 10000
        if index < len(model.data):model.simulate_individual(index)
        else:model.simulate_generic()
        
    def run_simulationCluster1LL(self, event):
        model = self.hsmmCluster1LL
        index = event.Id - 10000
        if index < len(model.data):model.simulate_individual(index)
        else:model.simulate_generic()
        
    def run_simulationCluster2HH(self, event):
        model = self.hsmmCluster2HH
        index = event.Id - 10000
        if index < len(model.data):model.simulate_individual(index)
        else:model.simulate_generic()
        
    def run_simulationCluster2MM(self, event):
        model = self.hsmmCluster2MM
        index = event.Id - 10000
        if index < len(model.data):model.simulate_individual(index)
        else:model.simulate_generic()
        
    def run_simulationCluster2LL(self, event):
        model = self.hsmmCluster2LL
        index = event.Id - 10000
        if index < len(model.data):model.simulate_individual(index)
        else:model.simulate_generic()
        
if __name__ == "__main__":
    app = HSMMSimDyadApp(False)
    app.MainLoop()

