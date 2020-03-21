import re

def match(info, name):
    if   info=='size':
        try:
            return re.search(r'\d+(x|X)\d+',name).group().replace('X', 'x')
        except:
            return ''
    if   info=='height':
        try:
            size = re.search(r'\d+(x|X)\d+',name).group().replace('X', 'x')
            return size.split('x')[0]
        except:
            return ''
    if   info=='width':
        try:
            size = re.search(r'\d+(x|X)\d+',name).group().replace('X', 'x')
            return size.split('x')[1]
        except:
            return ''
    elif info=='date':
        try:
            if re.search(r'\d{4}(-|_)\d{1}(-|_)\d{2}',name) is not None:
                date = re.search(r'\d{4}(-|_)\d{1}(-|_)\d{2}',name).group()
                date = date[:5] + '0' + date[5:]
            else:
                date = re.search(r'\d{4}(-|_)\d{2}(-|_)\d{2}',name).group()
            return date.replace('_', '-')
        except:
            return ''
    elif info=='color':
        try:
            return re.search(r'(RED)|(WHITE)|(GRAY)|(BLUE)|(GREEN)',name).group()
        except:
            return ''
    elif info=='model':
        try:
            if re.search(r'STELVIO Q4', name) is not None:
                return 'STELVIO Q4'
            else:
                return re.search(r'(GIULIA)|(GIULIETTA)|(STELVIO)|(MITO)',name).group()
        except:
            return ''
        
def are_referring_same_banner(ad_name, banner_name):
    if match('date', ad_name)   != match('date', banner_name):
        return False
    if match('color', ad_name)  != match('color', banner_name):
        return False
    if match('model', ad_name)  != match('model', banner_name):
        return False
    if match('height', ad_name) != match('height', banner_name):
        return False
    if match('width', ad_name)  != match('width', banner_name):
        return False 
    return True