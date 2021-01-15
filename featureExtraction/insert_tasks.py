from featureExtraction.utils import pss_extraction_utils_updated
from featureExtraction.utils import pss_extraction_utils_updated, taskqueue_utils

Obj = taskqueue_utils.create_proc_obj (config_file,cell_id)
unprocessed_inds = pss_extraction_utils_updated.findunprocessed_indices(Obj)
#tq = LocalTaskQueue(parallel=45) # use 5 processes
tq = TaskQueue('sqs://queue-name', region_name="us-east1-b", green=False)
tasks = (partial(pss_extraction_utils_updated.myprocessingTask,Obj,l,q) for q in unprocessed_inds)
tq.insert_all(tasks) 
