import win32com.client
import re


def get_usb_dev():
    default_dev =[]
    wmi = win32com.client.GetObject ("winmgmts:")
    for usb in wmi.InstancesOf ("win32_usbcontrollerdevice"):
        if ("VID_1E4E&PID_0100" in usb.Dependent or "VID_05A3&PID_8830" in usb.Dependent) and not "MI_00"in usb.Dependent:
        #if "USB" in usb.Dependent:
            result = re.search('DeviceID="USB(.+?)"', usb.Dependent)
            #result.group(1).split('\\\\')[2]
            
            #default_dev.append((result.group(1).split('\\\\')[1],result.group(1).split('\\\\')[2]))
            default_dev.append(result.group(1).split('\\\\')[2])
            #if "VID_1E4E&PID_0100" in usb.Dependent or "VID_05A3&PID_8830" in usb.Dependent:
            #if ""
            #print(result.group(1))
             
    return default_dev
#input("Press Enter to continue...")

#for dev  in default_dev:
#    print(dev[1])
#for dev  in get_usb_dev():
#    print(dev[1])
'''
wmi = win32com.client.GetObject ("winmgmts:")
for usb in wmi.InstancesOf ("win32_usbcontrollerdevice"):
    #if 
    if "USB" in usb.Dependent:
        result = re.search('DeviceID="USB(.+?)"', usb.Dependent)
        #default_dev.append((result.group(1).split('\\\\')[1],result.group(1).split('\\\\')[2]))
        #if "VID_1E4E&PID_0100" in usb.Dependent or "VID_05A3&PID_8830" in usb.Dependent:
        #if ""
        print(result.group(1))


print(default_dev)
usblist = wmi.InstancesOf ("win32_usbcontrollerdevice")
for dev  in default_dev:
#for usb in wmi.InstancesOf ("win32_usbcontrollerdevice"):
    #default_dev.append(usb.Dependent)
    #if "VID_1E4E&PID_0100" in usb.Dependent or "VID_05A3&PID_8830" in usb.Dependent:
        print (dev[1])
        if not dev[0] in usblist:
            
            print(usb.Dependent)
'''
