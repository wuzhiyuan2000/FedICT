import utils   
def get_loss_kd_server(method_name,args,client_index=None):
    if method_name=='lga_fd_balance':
        return utils.LKA_balance(args.temperature,args.dist_global,args.dist_locals[client_index],args.lka_U)
    elif method_name=='lga_fd_sim':
        return utils.LKA_sim(args.temperature,args.dist_global,args.dist_locals[client_index])
    else:
        raise Exception("method error")



def get_loss_kd_client(method_name,args,client_index):
    if method_name in ['lga_fd_balance','lga_fd_sim']:
        return utils.FPKD(args.temperature,args.dist_locals[client_index],args.fpkd_T)
    else:
        return utils.KL_Loss(args.temperature)
