import base64
from Crypto.Cipher import AES
from Crypto import Random
import random, string
import win_usb_api as usb

def AES_Decrypt( data):
    #vi = '496e7369676874566973696f6e'
    vi = 'InsightVision'
    key = '416c696365'
    pad = lambda s: s + (16 - len(s)%16) * chr(16 - len(s)%16)
    vic = pad(vi)
    key = pad(key)
    
    data = data.encode('utf8')
    print (data)
    encodebytes = base64.decodebytes(data)
    #encodebytes = decodebytes(data)
    # 將加密數據轉換位bytes類型數據
    cipher = AES.new(key.encode('utf8'), AES.MODE_CBC, vic.encode('utf8'))
    text_decrypted = cipher.decrypt(encodebytes)
    unpad = lambda s: s[0:-s[-1]]
    text_decrypted = unpad(text_decrypted)
    # 去補位
    text_decrypted = text_decrypted.decode('utf8')
    
    devs = usb.get_usb_dev()
    
    if devs[0] in text_decrypted and devs[1] in text_decrypted:
        return True
    else:
        return False
     