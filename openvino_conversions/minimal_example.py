

import openvino
import openvino.inference_engine
from openvino.inference_engine import IECore
ie = IECore()
# ie.unregister_plugin('MYRIAD') # doesnt help
devs = ie.available_devices
print(devs)

