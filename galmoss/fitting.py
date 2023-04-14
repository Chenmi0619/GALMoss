from ellip import Ellip
from sersic import Sersic
from parameters import Centre_x, Centre_y, Inclination, Axis_ratio, Effective_radius, Intensity, Sersic_index
from data import Data_Box
from collections.abc import Sequence as Seq
from typing import Union, Tuple, Optional
import numpy as np
import torch
import time
from functools import wraps
from tqdm import tqdm
import torch_optimizer as optim

from ellip import Ellip
import torch.utils.data as Data
from data import Imaging, Profiles, Data_Box
from astropy import convolution as conv
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# class Basic_fitting():
#   """
#   Base class for Parameters

#   Parameters
#   ----------
#   func
#       A function which returns the parameters need to be load in the optimism.

#   Returns
#   -------
#       A function that returns the scale-parameters need to be load in the optimism.
#   """
#   def __init__(
#     self
#   ):

#     """
#     The parameter class.

#     Parameters
#     ----------
#     parameters
#         The input parameters.
#     scale
#         If True, then will reflect the parameters into 0-1 field and use tanh function to scale the parameters inside this scale.
#     if_log
#         Some parameters will fit better in exp10 field.
#     if_fit
#         If this parameters need to be optimistic or just keep stable.
#     """    
  
#   def get_optim():
    
    
class Fitting():
  """
  Base class for Parameters

  Parameters
  ----------
  func
      A function which returns the parameters need to be load in the optimism.

  Returns
  -------
      A function that returns the scale-parameters need to be load in the optimism.
  """
  def __init__(
      self, 
      dataset: Data_Box,
      batch_size: int,
      iteration: int,
      optimizer = optim.DiffGrad
      
  ):
    self.dataset = dataset
    self.batch_size = batch_size
    self.iteration = iteration
    self.optimizer = optimizer
    
    self.sub_factor = 1
    self.grid = 0
    self.model_square = None
    self.model_jacco = None

  def __optim_list(self, data):
      print("using!")
      optim_list = []
         
      ams=0   
      for param in self.dataset.Profiles.flex.values():
        optim_data = {}
        _data = data[:, ams]
        _data.requires_grad=True
        optim_data['params'] = _data
        optim_data['lr'] = param.step_length
        optim_list.append(optim_data)
        ams+=1
      return optim_list
  
  def subsampling(self, sub_factor):
    self.sub_factor = sub_factor
       
  def modeling(self):
      ams=0
      for key in self.dataset.Profiles.profile_dict:
        profile = self.dataset.Profiles.profile_dict[key]
        aa = self.model_square.clone()
        aa[ams]= profile.image_2d_grid_from_(self.grid)
        ams+=1
        return aa
  
  def make_uncertainty(self, model, sigma, optim_list):
      ams=0
      param = self._make_param(optim_list, 0)
      param1 = param[:, 0]
      print(param1)
      # param1 = tuple([param[:, i] for i in range(len(param[0]))])
      
      mtot = self.dataset.Profiles.serssic.intensity.update_value
      print(model[0][0][0:5])
      
      mtot = self.dataset.Profiles.serssic.intensity.update_value
      mtot = torch.unsqueeze(torch.unsqueeze(mtot, dim=1), dim=1)
      re = self.dataset.Profiles.serssic.effective_radius.update_value
      re_ =  torch.unsqueeze(torch.unsqueeze(re, dim=1), dim=1)
      print(np.shape(re))
      q = self.dataset.Profiles.serssic.axis_ratio.update_value
      q = torch.unsqueeze(torch.unsqueeze(q , dim=1), dim=1)
      bn = self.dataset.Profiles.serssic.sersic_constant
      bn = torch.unsqueeze(torch.unsqueeze(bn, dim=1), dim=1)
      bn__n = self.dataset.Profiles.serssic.sersic_constant_gradient
      bn__n = torch.unsqueeze(torch.unsqueeze(bn__n, dim=1), dim=1)
      n = self.dataset.Profiles.serssic.sersic_index.update_value
      print(n)
      n = torch.unsqueeze(torch.unsqueeze(n, dim=1), dim=1)
      # n1 = n.clone()
      # n1.requires_grad=True
      r = self.dataset.Profiles.serssic.make_radius(self.grid)
      print("r", np.shape(r))
      pa = self.dataset.Profiles.serssic.inclination.update_value * torch.pi / 180
      pa = torch.unsqueeze(torch.unsqueeze(pa , dim=1), dim=1)
      xc = self.dataset.Profiles.serssic.centre_x.update_value
      xc = torch.unsqueeze(torch.unsqueeze(xc , dim=1), dim=1)
      yc = self.dataset.Profiles.serssic.centre_y.update_value
      yc = torch.unsqueeze(torch.unsqueeze(yc , dim=1), dim=1)
      print(np.shape(bn), np.shape(re_), np.shape(n), np.shape(bn__n))
      
      
      # 从这里开始注释的
      
      # start = time.time()
      # jac_tuple = torch.autograd.functional.jacobian(self.for_jaccobian1,
      #                                              inputs=param1,
      #                                              vectorize=True)
      # end = time.time()
      # print(jac_tuple)
      # print("time", end-start)
      ww = (1/(torch.pow(sigma[0], 2))).view(16384)
      www = torch.eye(16384).to(device) * ww
      # jtj = (jac_tuple.T.matmul(www)).matmul(jac_tuple)
      # print(jac_tuple)
      # print(np.shape(jac_tuple))
      
      # print(torch.pow(sigma[0], 2))
      # print(ww)
      # print(www)
      # print(jtj)
      shape = np.shape(model)
      b = model[0]
      b = b.view((shape[1] * shape[2]))
      print(b[0:5])
    
      print("--------------------")
      
      
      # """
      # mtot
      # """
      # model[0][0][0:5] * torch.log(torch.Tensor([10])).to(device)* -0.4 * 6.5 
      
      """
      n
      """     
      # bn_n gam__ bn_2n__检查正确
      A =  torch.pow(10, (-0.4 * mtot)) / (torch.pow(re_, 2) * 2 * np.pi * q)
      print("A1", A)
      gam = torch.exp(torch.lgamma(2 * n))
      # gam__ =  torch.autograd.grad(torch.exp(torch.lgamma(2 * n)), n, create_graph=True)[0]
      # gam__ = torch.autograd.grad(torch.exp(torch.lgamma(2 * 4.95*(n1))), n1, create_graph=True)[0]
      # gam__ = torch.unsqueeze(torch.unsqueeze(gam__, dim=1), dim=1)
      gam__ =  (torch.exp(torch.lgamma(2 *  (n+ 1e-4)))  - torch.exp(torch.lgamma(2 *n))) * 4.95 / 1e-4
      print(gam__)
      gam_bn_2n = torch.pow(10, torch.log10(gam) - 2 * n * torch.log10(bn)) 
      bn_2n = torch.pow(10, - 2 * n * torch.log10(bn)) 
      bn_2n__ = bn_2n * (-2 * ( torch.log(bn) * 4.95 + n * (bn__n / bn)))
      
      B = (gam_bn_2n * n * torch.exp(bn)) ** 2
      

      C__ = torch.exp(-bn * (torch.pow(r / re_, (1.0 / n)) - 1)) * ( - (bn__n * (torch.pow(r / re_, (1.0 / n)) - 1)  + bn * (torch.log(r / re_) * torch.pow(r / re_, (1.0 / n)) * (-1 / torch.pow(n, 2)) * 4.95)
                                                                        ))
      C = torch.exp(-bn * (torch.pow(r / re_, (1.0 / n)) - 1))
      
      D__ = gam__ *  (bn_2n * n * torch.exp(bn)) + gam * (bn_2n__ * n * torch.exp(bn) + bn_2n * (torch.exp(bn) * 4.95 + (n*torch.exp(bn) * bn__n)))
      D = gam_bn_2n * n * torch.exp(bn) 
      
      a_b = torch.unsqueeze(torch.unsqueeze((A/B), dim=1), dim=1)
      print(np.shape(A), np.shape(B))
     
      n_gra = (A/B) * (C__ * D - C * D__)
      
      
      n_gra_input_psf = n_gra.unsqueeze(0).to(dtype=torch.float)
      # print("n_gra", np.shape(n_gra_input_psf))
      # self.weight_input_psf = psfim_data.unsqueeze(1)
  #         # print(np.shape(antisky_model_after_psf))
      aaa = torch.sum(self.weight_input_psf[0], dim=[1, 2])
      n_gra_after_psf = F.conv2d(n_gra_input_psf,self.weight_input_psf,padding="same", groups=len(self.galaxy_train_index)) #/ aaa
      n_gra_after_psf = torch.squeeze(n_gra_after_psf)
      aaa = torch.unsqueeze(torch.unsqueeze(aaa, 1), 1)
      n_gra_after_psf = n_gra_after_psf/aaa

      n_gra1 = n_gra_after_psf
    
      """
      re
      """     
      E = torch.pow(10, (-0.4 * mtot)) / (gam_bn_2n * n * 2 * np.pi * q * torch.exp(bn) * torch.pow(re_, 4))
      F__ = torch.exp(-bn * (torch.pow(r / re_, (1.0 / n)) - 1)) * (-bn) * ((1/n) * torch.pow((r / re_), (1 / n -1)) * ((- r * 25.126262626262623)/re_**2))
      Ff = torch.exp(-bn * (torch.pow(r / re_, (1.0 / n)) - 1))
      G__ = 2 * re_ * 25.126262626262623
      G = re_ ** 2
      
      re_gra = E * (F__ * G - Ff* G__)
      
      re_gra_input_psf = re_gra.unsqueeze(0).to(dtype=torch.float)
      print("re_gra", np.shape(re_gra_input_psf))
      aaa = torch.sum(self.weight_input_psf[0], dim=[1, 2])
      re_gra_after_psf = F.conv2d(re_gra_input_psf,self.weight_input_psf,padding="same", groups=len(self.galaxy_train_index)) #/ aaa
      re_gra_after_psf = torch.squeeze(re_gra_after_psf)
      aaa = torch.unsqueeze(torch.unsqueeze(aaa, 1), 1)
      re_gra_after_psf = re_gra_after_psf/aaa

      re_gra1 = re_gra_after_psf
      
      """
      q
      """ 
      
      x = self.grid[0].to(device) - xc
      y = self.grid[1].to(device) -yc
      
      x_maj = torch.cos(pa) * x + torch.sin(pa) * y
      x_min = -x * torch.sin(pa) + y * torch.cos(pa)
      
      r2q = 0.5 * torch.pow(x_maj**2 + (x_min / q)**2, -0.5) * ((-x_min**2 * 2 * 0.495) / (q ** 3))
      H = torch.pow(10, (-0.4 * mtot)) / (gam_bn_2n * n * 2 * np.pi * torch.exp(bn) * torch.pow(re_, 2) * torch.pow(q, 2))
      I__ = torch.exp(-bn * (torch.pow(r / re_, (1.0 / n)) - 1)) * (-bn) * ((1/n) * torch.pow(r/re_, 1/n -1) * (1/re_) * r2q)
      I = torch.exp(-bn * (torch.pow(r / re_, (1.0 / n)) - 1))
      J__ = 0.495
      J = q
      q_gra = H * (I__ * J - I * J__)
      
      q_gra_input_psf = q_gra.unsqueeze(0).to(dtype=torch.float)
      
      

      # self.weight_input_psf = psfim_data.unsqueeze(1)
  #         # print(np.shape(antisky_model_after_psf))
      aaa = torch.sum(self.weight_input_psf[0], dim=[1, 2])
      q_gra_after_psf = F.conv2d(q_gra_input_psf,self.weight_input_psf,padding="same", groups=len(self.galaxy_train_index)) #/ aaa
      q_gra_after_psf = torch.squeeze(q_gra_after_psf)
      aaa = torch.unsqueeze(torch.unsqueeze(aaa, 1), 1)
      q_gra_after_psf = q_gra_after_psf/aaa

      q_gra1 = q_gra_after_psf

      """
      pa
      """
      pa__ = 90 * np.pi / 180
      xmaj2pa = -torch.sin(pa) * x * pa__ + torch.cos(pa) * y * pa__
      xmin2pa = -torch.cos(pa) * x * pa__ - torch.sin(pa) * y * pa__
      r2pa = 0.5 * torch.pow(x_maj**2 + (x_min / q)**2, -0.5) * (2 * x_maj * xmaj2pa + (2 * x_min * xmin2pa) / q**2)
  #     K = torch.pow(10, (-0.4 * mtot)) / (gam_bn_2n * n * 2 * np.pi * torch.exp(bn) * torch.pow(re_, 2) * q)
  #     L__ = torch.exp(-bn * (torch.pow(r / re_, (1.0 / n)) - 1)) * (-bn) * ((1/n) * torch.pow(r/re_, 1/n -1) * (1/re_) * r2pa)

  #     pa_gra = K * L__
      
      
  #     pa_gra_input_psf = pa_gra.unsqueeze(0).to(dtype=torch.float)
      
      

  #     # self.weight_input_psf = psfim_data.unsqueeze(1)
  # #         # print(np.shape(antisky_model_after_psf))
  #     aaa = torch.sum(self.weight_input_psf[0], dim=[1, 2])
  #     pa_gra_after_psf = F.conv2d(pa_gra_input_psf,self.weight_input_psf,padding="same", groups=len(self.galaxy_train_index)) #/ aaa
  #     pa_gra_after_psf = torch.squeeze(pa_gra_after_psf)
  #     aaa = torch.unsqueeze(torch.unsqueeze(aaa, 1), 1)
  #     pa_gra_after_psf = pa_gra_after_psf/aaa

  #     pa_gra1 = pa_gra_after_psf
      pa_gra2 = model* (-bn) * ((1/n) * torch.pow(r/re_, 1/n -1) * (1/re_) * r2pa)
      # print("r2pa1", r2pa)
       
      """
      x
      """
      xmaj2xc = torch.cos(pa) * (-10)
      xmin2xc = -torch.sin(pa) * (-10)
      
      xmaj2yc = torch.sin(pa) * (-10)
      xmin2yc = torch.cos(pa) * (-10)      
      r2xc = 0.5 * torch.pow(x_maj**2 + (x_min / q)**2, -0.5) * (2 * x_maj * xmaj2xc + (2 * x_min * xmin2xc) / q**2)
      r2yc = 0.5 * torch.pow(x_maj**2 + (x_min / q)**2, -0.5) * (2 * x_maj * xmaj2yc + (2 * x_min * xmin2yc) / q**2)
      xc_gra1 = model* (-bn) * ((1/n) * torch.pow(r/re_, 1/n -1) * (1/re_) * r2xc)
      yc_gra1 = model* (-bn) * ((1/n) * torch.pow(r/re_, 1/n -1) * (1/re_) * r2yc)


      

      
      
      # print(n_gra1[0, 0, 0:5])
      # print(np.shape(jac_tuple))
      # print(jac_tuple[0:5, 6])
      # print("re_gra", np.shape(re_gra1))
      # print("n_gra1", np.shape(n_gra1))
      # print(jac_tuple[0:50,5] / (model[0][0][0:50] * torch.log(torch.Tensor([10])).to(device)* -0.4 * 6.5))
      
      # print(jac_tuple[0:50,6] / (n_gra1[0][0][0:50]))
      # print(jac_tuple[0:50,4] / (re_gra1[0][0][0:50]))
      # print(jac_tuple[0:50,3] / (q_gra1[0][0][0:50]))
      # print(jac_tuple[0:50,2] / (pa_gra2[0][0][0:50]))
      # print(jac_tuple[0:50,1] / (yc_gra1[0][0][0:50]))
      # print(jac_tuple[0:50,0] / (xc_gra1[0][0][0:50]))

      
      # print(model[0][0][0])
      # print(model[0][0][0] * torch.log(torch.Tensor([10])).to(device)* -0.4 * 6.5 ) 
      # print(jac_tuple[0,5])
      
      
      # JTJ_LU = torch.lu(jtj)
      
      # JTJ_inv = torch.lu_solve(torch.eye(7).to(device), *JTJ_LU)
      
      # fi = torch.pow(torch.diag(JTJ_inv).to(device), 0.5)
      # print("fi", fi)
      # print(self.dataset.Profiles.serssic.intensity.update_value)
      grad_ser = self.dataset.Profiles.serssic.grad_from(self.grid, torch.squeeze(self.weight_input_psf), self.galaxy_train_index)
      a = grad_ser.Conv_center_x().view(len(self.galaxy_train_index), 1, -1)[0]
      b = grad_ser.Conv_center_y().view(len(self.galaxy_train_index), 1, -1)[0]
      c = grad_ser.Conv_inclination().view(len(self.galaxy_train_index), 1, -1)[0]
      d = grad_ser.Conv_axis_ratio().view(len(self.galaxy_train_index), 1, -1)[0]
      e = grad_ser.Conv_effective_radius().view(len(self.galaxy_train_index), 1, -1)[0]
      f = grad_ser.Conv_sersic_index().view(len(self.galaxy_train_index), 1, -1)[0]
      g = grad_ser.Conv_intenisty().view(len(self.galaxy_train_index), 1, -1)[0]
      print(np.shape(a))
      J = torch.concat((a, b, c, d, e, f, g), dim=0).T
      print(np.shape(aaa))
      print(J)
      JTJ = (J.T.matmul(www)).matmul(J)
      JTJ_LU = torch.lu(JTJ)
      
      JTJ_inv = torch.lu_solve(torch.eye(7).to(device), *JTJ_LU)
      
      fi = torch.pow(torch.diag(JTJ_inv).to(device), 0.5)
      print("fi", fi)
      # print(
      #       self.galaxy_train_index,
      #       self.dataset.Profiles.serssic.effective_radius.sq_V* 0.396,
      #       self.dataset.Profiles.serssic.sersic_index.sq_V ,
      #       self.dataset.Profiles.serssic.intensity.sq_V + 22.5,
      #       self.dataset.Profiles.serssic.axis_ratio.sq_V)
      print(self.dataset.Profiles.serssic.inclination.uncertainty,
            self.dataset.Profiles.serssic.inclination.update_value,
            self.dataset.Profiles.serssic.inclination.fit_value)
      # print(a[0][0][0:50] / (xc_gra1[0][0][0:50]))
      # print(b[0][0][0:50] / (yc_gra1[0][0][0:50]))
      # print(c[0][0][0:50] / (pa_gra2[0][0][0:50]))
      # print(d[0][0][0:50] / (q_gra1[0][0][0:50]))
      # print(e[0][0][0:50] / (re_gra1[0][0][0:50]))
      # print(f[0][0][0:50] / (n_gra1[0][0][0:50]))
      # print(g[0][0][0:50] / ((model[0][0][0:50] * torch.log(torch.Tensor([10])).to(device)* -0.4 * 6.5)))

  def for_jaccobian(self, *args):
    
    b = torch.Tensor([]).to(device)
    for i in np.arange(len(args)):
      data = torch.unsqueeze(args[i], dim=0)
      b = data if len(b) == 0 else torch.concat((b, data), dim=0)

    ams=0
    for param_class in self.dataset.Profiles.flex.values():
      data = b[:, ams]
      param_class.update(data)
      ams = ams +1
    
    a = self.modeling()
    shape = np.shape(a)
    b = torch.sum(a, dim=(0, 1))
    b = b.view((shape[2] * shape[3]))

    # b = a[0, 1].view((shape[2] * shape[3]))
    # print(np.shape(a))

    return b 
  
  
  def for_jaccobian1(self, param):
    
    ams=0
    for param_class in self.dataset.Profiles.flex.values():
      data = param[ams]
      param_class.update(data)
      ams = ams +1
    
    a = self.modeling()
    shape = np.shape(a)
    # b = a[0, 0]
    # b = b.view((shape[2] * shape[3]))
    
  #   antisky_model = a.sum(axis=0)
    
#       antisky_model = ori_model.sum(axis=0)
#       start21 = time.time()export LD LIBRARY PATH=SLD LIBRARY PATH:/usr/local/cfitsio-3.47
#       # 这里添加一层的意思是单通道
#       if psfim_data.sum() == 0:
#         antisky_model_after_psf = antisky_model
#       else:
    antisky_model_input_psf = antisky_model.unsqueeze(0).to(dtype=torch.float)

    # self.weight_input_psf = psfim_data.unsqueeze(1)
#         # print(np.shape(antisky_model_after_psf))
    aaa = torch.sum(self.weight_input_psf[0], dim=[1, 2])
    antisky_model_after_psf = F.conv2d(antisky_model_input_psf,self.weight_input_psf,padding="same", groups=len(self.galaxy_train_index)) #/ aaa
    antisky_model_after_psf = torch.squeeze(antisky_model_after_psf)
    aaa = torch.unsqueeze(torch.unsqueeze(aaa, 1), 1)
    antisky_model_after_psf = antisky_model_after_psf/aaa

    model = antisky_model_after_psf
    model = model[0]
    model = model.view((shape[2] * shape[3]))
    # b = a[0, 1].view((shape[2] * shape[3]))
    # print(np.shape(a))

    return model 
      

  def make_grid(self):
      x = torch.Tensor(list(range(1, (self.dataset.data_size[1])+1))).to(device)
      y = torch.flip(torch.Tensor(list(range(1, (self.dataset.data_size[0])+1))), [0]).to(device)
      xy = torch.meshgrid(y, x)
      
      self.grid = xy

      # return xy
  
  
  def _make_param(self, optim_list, m):
    total = torch.Tensor([]).to(device)
    # total = tuple([optim_list[ams]['params'][0][m] for ams in range(len(self.dataset.Profiles.flex))])
    for ams in np.arange(len(self.dataset.Profiles.flex)):
      data = torch.unsqueeze(optim_list[ams]['params'][0], dim=0)
      total = data if len(total) == 0 else torch.concat((total, data), dim=0)
    return total
  
  def _back_param(self, optim_list):
    ams=0
    for param in self.dataset.Profiles.flex.values():
      data = optim_list[ams]['params'][0]
      ams += 1
      param.update(data)
  
  
  def make_sigma(self, sigma, scale, mask):
    aa = sigma * scale[:, 0] * mask
    return aa
  
  def out_param(self):
    # 
    for key in self.dataset.Profiles.profile_dict:
      profile = self.dataset.Profiles.profile_dict[key]
      grad = profile.grad_from(self.grid, torch.squeeze(self.weight_input_psf), self.galaxy_train_index)
      grad.grad_backward()
    
    J = torch.Tensor([]).to(device)
    for param in self.dataset.Profiles.flex.values():
      param_j = param.grad.view(len(self.galaxy_train_index), 1, -1)
      if len(J) == 0:
        J = param.grad.view(len(self.galaxy_train_index), 1, -1)
      else:
        J = torch.concat((J, param_j), dim=1)
    
    _weight = (1/(torch.pow(self.sigma_data, 2))).view(len(self.galaxy_train_index), 1, -1)
    weight_eye = torch.eye(len(_weight[0, 0])).to(device).repeat(len(self.galaxy_train_index), 1, 1)
    weight = weight_eye * _weight

    # JTJ = (J.matmul(weight)).matmul(J.T) # 10, 7, 7
    JTJ = torch.matmul(torch.matmul(J, weight), J.transpose(1, 2))
    JTJ_LU = torch.lu(JTJ)

    t = torch.eye(len(self.dataset.Profiles.flex)).to(device)
    t = t.repeat(len(self.galaxy_train_index), 1, 1)

    JTJ_inv = torch.lu_solve(t, *JTJ_LU) # 10, 7, 7
    
    fi = torch.pow(torch.diagonal(JTJ_inv, dim1=-2, dim2=-1), 0.5) # 10, 7 

    ams = 0
    for param in self.dataset.Profiles.flex.values():
      param.uncertainty_store(fi[:, ams]) # 10, 1
      
      param.value_store(param.update_value)
      ams = ams + 1
      param.grad_clean()
    
    
    
    
    
      
    
    
  def train(self):
    b = Data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)
    self.make_grid()
    for self.galaxy_train_index,fits_data, segim_data, mskim_data, psfim_data, self.sigma_data, flex_data, fix_data in b:
      self.model_square = torch.zeros((len(self.dataset.Profiles.profile_dict), 
                                len(self.galaxy_train_index), 
                                self.dataset.data_size[0], 
                                self.dataset.data_size[1])).to(device)
      self.model_jacco = torch.zeros((len(self.dataset.Profiles.profile_dict), 
                                len(self.galaxy_train_index), 
                                self.dataset.data_size[0] * self.dataset.data_size[1])).to(device)
      _optim_list = self.__optim_list(flex_data)
      optimizerrr = self.optimizer(_optim_list, eps=1e-7) 
      
      for iter in tqdm(np.arange(self.iteration)):
        
        self._back_param(_optim_list)
        
        aa = self.modeling()
  #       ori_model, sky, r_total = modeling(config, optim_list, normal_list, param["info"], model_square, scale_bias)
        antisky_model = aa.sum(axis=0)
        
  #       antisky_model = ori_model.sum(axis=0)
  #       start21 = time.time()export LD LIBRARY PATH=SLD LIBRARY PATH:/usr/local/cfitsio-3.47
  #       # 这里添加一层的意思是单通道
  #       if psfim_data.sum() == 0:
  #         antisky_model_after_psf = antisky_model
  #       else:
        antisky_model_input_psf = antisky_model.unsqueeze(0).to(dtype=torch.float)
        self.weight_input_psf = psfim_data.unsqueeze(1)
  #         # print(np.shape(antisky_model_after_psf))
        aaa = torch.sum(psfim_data, dim=[1, 2])
        antisky_model_after_psf = F.conv2d(antisky_model_input_psf,self.weight_input_psf,padding="same", groups=len(self.galaxy_train_index)) #/ aaa
        antisky_model_after_psf = torch.squeeze(antisky_model_after_psf)
        aaa = torch.unsqueeze(torch.unsqueeze(aaa, 1), 1)
        antisky_model_after_psf = antisky_model_after_psf/aaa

        model = antisky_model_after_psf
        

        chi_1 = torch.pow((fits_data - model).to(device)* segim_data[:, 0], 2) / torch.pow(self.sigma_data, 2)
        scale_dim = torch.sum(segim_data[:, 0], dim=[1, 2]) - 7
        loss = torch.sum(torch.sum(chi_1, dim=[1, 2]) / scale_dim )
        # print(loss)

        
        
        optimizerrr.zero_grad()
        if iter == self.iteration-1:
          loss.backward(retain_graph=True)
        else:
          loss.backward()
        optimizerrr.step()
        

        # print(loss)

    #       ams_loss.append((loss/len(galaxy_index)).detach().cpu().numpy())
    #       end = time.time()
    #       torch.cuda.empty_cache()
      # mm = self.modelll(self.grid)
      # kk = self.modeling(self.grid)
      bb =(( (fits_data - model).to(device)* segim_data[:, 0] )/ self.sigma_data) * (-2)
      loss2 = torch.Tensor([])
      loss2 = torch.sum(model)

      loss.detach()
      print(loss)
      aaa = self.make_sigma(self.sigma_data, segim_data, mskim_data)
      self.out_param()
      self.make_uncertainty(model, self.sigma_data, _optim_list)
      
      model1 = model.detach().cpu().numpy()
      res = fits_data - model
      res = res.detach().cpu().numpy()
      fits1 = fits_data.detach().cpu().numpy()
      np.save("model1.npy", model1)
      np.save("res.npy", res)
      np.save("fits1.npy", fits1)
      
      
      
      
      
      
    


# if __name__ == "__main__":
#   data_shape = [10, 10]
#   x = torch.Tensor(list(range(1, (data_shape[1])+1)))
#   y = torch.flip(torch.Tensor(list(range(1, (data_shape[0])+1))), [0])
#   xy = torch.meshgrid(y, x)
  
#   center_x = Centre_x(parameters=[5])
#   center_y = Centre_y(parameters=[5])
#   incli = Inclination(parameters=100)
#   axi = Axis_ratio(parameters=0.2)
#   effr = Effective_radius(parameters=5)
#   index = Sersic_index(parameters=3)
#   inten = Intensity(parameters=16)
  
#   test = Sersic(centre_x= center_x, centre_y= center_y, inclination=incli, axis_ratio= axi, effective_radius=effr,
#                 sersic_index=index, intensity=inten)

#   aaa = Data_Box(galaxy_index=["1-76"],
#                  image_path=["/data/public/ChenM/MIFIT/MANGA/data/test/fitim"],
#                  mask_path=["/data/public/ChenM/MIFIT/MANGA/data/test/mskim"],
#                  psf_path=["/data/public/ChenM/MIFIT/MANGA/data/test/psfim"],
#                  sigma_path=["/data/public/ChenM/MIFIT/MANGA/data/test/sigma/"],
#                  seg_path=["/data/public/ChenM/MIFIT/MANGA/data/test/segim"])
  
#   aaa.from_profiles(Profiles(serssic = test))
  
#   fitting = Fitting(dataset=aaa, batch_size=1, iteration=100, optimizer=optim.DiffGrad)
#   fitting.train()

