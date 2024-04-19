import numpy as np
import torch
import torch.utils.data as Data
from torch.func import vmap, jacrev
from tqdm import tqdm


class Bootstrap:
    """
    Class for uncertainty calculation method Bootstrap.
    """ 
    def prepair_data(self, center_fix: bool):
        """
        Refitting in bootstrap process often meets challenge if the
        pixels near the galaxy center is not be resampled. This function 
        is for changing the fix statement of center_x and center_y. If 
        center_fix == True, the profile center will be fixed, which will
        help to make the refitting process more stable.

        Parameters
        ----------
        center_fix
            Bool augument, to determin the center parameters will be fixed
            or not.
        """
        if center_fix == True:
            self.dataset.Profiles.bsp_variableParam = {}
            self.dataset.Profiles.bsp_fixedParam = {}
            for param_name, param in self.dataset.Profiles.variableParam.items():
                key = param.param_name
                if key == "cen_x" or key == "cen_y":
                    param.fit = False
                    self.dataset.Profiles.bsp_fixedParam[param_name] = param
                else:
                    self.dataset.Profiles.bsp_variableParam[param_name] = param  
            for param in self.dataset.Profiles.fixedParam.values():
                self.dataset.Profiles.bsp_fixedParam[param_name] = param    

    def resample_mask(self, mask_data: torch.tensor):
        """
        this function is for resampling the pixels of galaxy image data. 
        
        Parameters
        ----------        
        mask_data:
            The mask data will be resampled.
        
        Returns
        -------
            The resampled mask data.
        """
        random_integers = torch.randint(0, 
                                        np.prod(mask_data.shape[1:]), 
                                        (np.prod(mask_data.shape[1:]), )
                                        )           
        flat_mask = mask_data.reshape(len(self.index), -1)
        new_mask = torch.zeros(np.shape(flat_mask)).to(self.dataset.device)
        unique_arr, counts = torch.unique(random_integers, return_counts=True)
        new_mask[:, unique_arr] = (flat_mask[:, unique_arr] 
                                   * counts.to(self.dataset.device))
        new_mask = new_mask.reshape(len(self.index), 
                                    np.shape(mask_data)[1], 
                                    np.shape(mask_data)[2])     
        return new_mask
             
    def calculate_uncertainty(self, P_dict):
        """
        Calculate the fitting uncertainty via calculating the variance 
        among the fitted value of parameters.
        """
        for param in P_dict.values():
            uncertainty = torch.std(param.bsp_best_value, dim=2)

            if param.dim == "1d":
                param.uncertainty = param.inv_from_mapping_uc(
                    uncertainty.reshape(param.param_length))
            
    def bootstrap(self, 
                  n_resample: int = 100, 
                  iteration: int = 100, 
                  center_fix: bool = True
        ):
        """
        Refit the resampled galaxies, as a way to calculating fitting 
        uncertianties.

        Parameters
        ----------
        n_resample
            The time of the resampling process. The suggest value is 100.
        iteration
            The maximum fitting iteration, the fitting process will be 
            shut down when reaching this number.
        center_fix
            To decide whether set the fit statement of paramter 
            (Center_x & Center_y) to False. The suggest value is True. 
        """
        self.dataset.mode = "bootstrap"
        self.prepair_data(center_fix)
        bsp_dataset = Data.DataLoader(dataset=self.dataset, 
                                      batch_size=self.batch_size, 
                                      shuffle=False)
        for (self.index, 
             self.data_cube, 
             self.psf, 
             param_final_value, 
             consant_value) in bsp_dataset:

            for _ in tqdm(np.arange(n_resample)):              
                optim_list = self.make_optim_list(
                    self.dataset.Profiles.bsp_variableParam, 
                    param_final_value)
                self.load_constant(self.dataset.Profiles.bsp_fixedParam, 
                                   consant_value)
                optimizer = self.optimizer(optim_list) 
                fitting_metrics = self.initialize_fitting_metrics()
                
                for iter in np.arange(iteration): 
                    self.refresh_variable(
                        self.dataset.Profiles.bsp_variableParam, 
                        optim_list)
                
                    conv_model = self.modeling(
                        self.dataset.Profiles.conv_profile_dict
                        )
                    no_conv_model = self.modeling(
                        self.dataset.Profiles.no_conv_profile_dict
                        )
                    model_tot = (self.add_convolution(conv_model.sum(axis=0)) 
                                 + no_conv_model.sum(axis=0))
                    
                    loss, chi_mu = self.optimizing(self.data_cube[:, 0], 
                                                    model_tot, 
                                                    self.data_cube[:, 1])  
                    loss.backward()
                    optimizer.step()  
                    optimizer.zero_grad()    

                    update, fitting_metrics = self.detect_update(
                        fitting_metrics,
                        chi_mu, 
                        iter
                    )  
                    if not update:
                        break    
                self.refitted_param_append(
                    self.dataset.Profiles.bsp_variableParam, 
                    sampling_idx = _
                    )


            self.refitted_param_append(self.dataset.Profiles.bsp_variableParam, 
                                      total=True)  
        self.calculate_uncertainty(self.dataset.Profiles.bsp_variableParam)

    def refitted_param_append(self, 
                            P_dict: dict, 
                            total:bool = False,
                            sampling_idx: int = 999):
        """
        This function is for storing the parameters refitting value inside 
        bootstrapping process. If total is True, which means the 
        process is appending batch refitted value to total refitted value.
        If total is False, which means the process is appending total 
        refitted value with the previous total refitted value (in principle
        each galaxy's each parameters will have n_sampled values)
        
        Parameters
        ----------
        sampling_idx
            The idx of sampling process, if zero, need to reset bsp_batch_best_value
        P_dict
            Dictionary of variable parameters
        total
            Determines to running which kind of appending process.
        """
        if not total:
            for param in P_dict.values():
                if (hasattr(param, 'bsp_batch_best_value') and sampling_idx != 0):
                    param.bsp_batch_best_value = torch.cat((
                        param.bsp_batch_best_value, 
                        param.batched_mapped_best_value.unsqueeze(2)), 
                        dim=2)     
                else:
                    param.bsp_batch_best_value = (param
                                                   .batched_mapped_best_value
                                                   .unsqueeze(2))      
        else:
            for param in P_dict.values():              
                if hasattr(param, 'bsp_best_value'):
                    param.bsp_best_value = torch.cat((
                        param.bsp_best_value, 
                        param.bsp_batch_best_value), 
                        dim=0)
                else:
                    param.bsp_best_value = param.bsp_batch_best_value


class CovarianceMatrix:
    """
    Class for uncertainty calculation method CovarianceMatrix.
    """    
    def CM_uncertainty_append(self, batched_uncertainty: torch.tensor):
        """
        Append the uncertainty values for each batch of galaxies after
        the completion of each uncertainty calculation process.

        Parameters
        ----------
        batched_uncertainty
            The uncertainty for each batch of galaxies.
        """          
        if hasattr(self, 'total_uncertainty'):
            self.total_uncertainty = torch.concat((self.total_uncertainty, 
                                                   batched_uncertainty), 
                                                  dim=0)                 
        else:
            self.total_uncertainty = batched_uncertainty                    
        
    def calculate_uncertainty_from_J(self, jacobian_matrix: torch.tensor):
        """
        Use Jacobian matrix to appraxmate Hessian matrix, then use the 
        diagnose of the inverse of Hessian matrix to calculate parameter 
        fitting uncertainty.

        Parameters
        ----------
        jacobian_matrix
            The corresponding jacobian matrix.

        Returns
        -------
        uncertainty
            The uncertainty calculated from Jacobian matrix.
        """           
        new_J, filtered_eyed_W = self.reconstruct(jacobian_matrix)
        jTj = torch.bmm(torch.bmm(new_J, filtered_eyed_W), 
                        new_J.transpose(1, 2))   
        LU, pivots = torch.linalg.lu_factor(jTj.detach())
        
        jTj_inv = torch.linalg.lu_solve(
            LU, pivots, 
            torch.eye(len(self.dataset.Profiles.variableParam))
            .to(self.dataset.device)
            .repeat(len(self.index), 1, 1)) 
        
        uncertainty = torch.pow(torch.diagonal(jTj_inv, dim1=-2, dim2=-1), 0.5)

        return uncertainty
            
    def store_uncertainty(self, param_value):
        """
        This function is responsible for separating and storing the 
        uncertainties associated with each fitting parameter in the 
        corresponding parameter object. It iterates through each parameter 
        involved in the fitting process, assigning the relevant uncertainty 
        values from the provided tensor.
        """
        for idx, param in enumerate(self.dataset.Profiles.variableParam
                                    .values()):
            param.uncertainty = param.inv_from_mapping_uc(self.total_uncertainty[:, idx])
                              
    def reconstruct(self, J:torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        """
        To extract the effective value from weight matrix and Jacobian 
        matrix based on the mask matrix, and resize to flat matrix.

        Parameters
        ----------
        J
            The original Jacobian matrix.
        Returns
        -------
        new_J
            The effective and flat Jacobian matrix.
        new_eyed_W
            The eye matrix, with values equals flat weight matrix.
        """
        flat_weight = ((1 / (torch.pow(self.data_cube[:, 1] , 2)))
                           .view(len(self.index), 1, -1))     
        if self.dataset.mask_path is not None:   
            reshape_mask = self.data_cube[:, 2].reshape(len(self.index), -1)
            eff_pix_lengths = torch.sum(reshape_mask, dim=1).int()

            new_J = torch.zeros(len(self.index), 
                                len(self.dataset.Profiles.variableParam), 
                                torch.sum(reshape_mask, dim=1).int().max(), 
                                dtype=torch.float32).to(self.dataset.device) 
            new_W = torch.zeros(len(self.index), 
                                1, 
                                torch.sum(reshape_mask, dim=1).int().max(), 
                                dtype=torch.float32).to(self.dataset.device)  

            for i in range(len(self.index)): 
                eff_counts = eff_pix_lengths[i].item()
                where_eff = torch.nonzero(reshape_mask[i] == 1, 
                                                   as_tuple=True)[0]
                new_W[i, 0, : eff_counts] = flat_weight[i, :, where_eff]
                new_J[i, :, : eff_counts] = J[i, :, where_eff]
            eye_size = int(torch.sum(reshape_mask, dim=1).max().item())        
                                    
        else:
            new_J = J
            new_W = flat_weight            
            eye_size = np.prod(self.data_cube[:, 1].shape[1:])
        
        eyes_mask = (torch.eye(eye_size)
                     .expand(len(new_J), 
                             eye_size, 
                             eye_size).to(self.dataset.device))
        new_eyed_W = eyes_mask * new_W.float() 
        
        return new_J, new_eyed_W
         
    def jacobian(self, param: torch.Tensor) -> torch.Tensor:
        """
        This function **would not** be used to calculate model value, 
        only to be load in jacrevm for calculating the Jacobian matrix. 
        
        Parameters
        ----------
        param
            The fitted parameters after fitting process.
        model
            The model image calculated by fitted parameters.
        """
        ams = 0
        for param_class in self.dataset.Profiles.variableParam.values():
            param_class.batched_updating_value = (
                param[ams: ams + param_class.dim2_length])
            ams += param_class.dim2_length
                
        conv_model = self.modeling(self.dataset.Profiles.conv_profile_dict, 
                                   mode="updating_model") 
        no_conv_model = self.modeling(self.dataset.Profiles.no_conv_profile_dict,
                                      mode="updating_model")    
        
        model_tot =( conv_model.sum(axis=0) + no_conv_model.sum(axis=0)).sum(axis=0)
        model = model_tot.view((self.dataset.data_size[0] 
                                * self.dataset.data_size[1]))
        return model        
       
    def covar_mat(self, bs: int = 2):
        """
        Calculate the fitting uncertainty for each parameters.

        Parameters
        ----------
        bs
            the batch-size for calculating covariance matrix. 
            Cause the huge calculation cost in Jacobian matrix,
            the bs here should be much smaller than the fitting
            batch size.
        """
        self.dataset.mode = "covar_mat"
        dataset = Data.DataLoader(dataset=self.dataset, 
                                  batch_size=bs, 
                                  shuffle=False)
    

        for (self.index, 
             self.data_cube, 
             param_final_value, 
             consant_value) in dataset:
            self.load_constant(self.dataset.Profiles.fixedParam, consant_value)
            with torch.no_grad():
                jacobian = vmap(jacrev(self.jacobian, argnums=0), 
                                in_dims=0)(param_final_value)
                J = torch.transpose(jacobian, 1, 2)
            self.CM_uncertainty_append(self.calculate_uncertainty_from_J(J))

        self.store_uncertainty(param_final_value)


class Uncertainty(Bootstrap, CovarianceMatrix):
    def __init__(self):  
        super().__init__() 

    def uncertainty(self, method = "covar_mat", *args, **kwargs):
        """
        Start calculating fitting uncertainty follows the method, and 
        resave the fitted value.
        """             
        if next(
            iter(self.dataset.Profiles.variableParam.values())
            ).total_mapped_best_value is not None: 
            if method == "covar_mat":
                self.covar_mat(*args, **kwargs)
            elif method == "bootstrap":
                self.bootstrap(*args, **kwargs)
            else:
                raise ValueError("Wrong method text, should be one of: "
                                 "covar_mat or bstrap")
        else:
            raise ValueError("There isn't any fitted value bein stored inside"
                             " each parameter object, please refit, or reload"
                             " the fitted value into the parameter object, and"
                             " then change the iteration number to zero!")

        if isinstance(self.dataset.result_path, str):
                self.save_param()   
                print("The parameter fitting result is resaved in {} "
                      "successfully with uncertainty calculated by {}!"
                      .format(self.dataset.img_block_path, method))  
        else:
            print("The parameter fitting result becuase result_path is not set.")   
 

