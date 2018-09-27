def mathematica2py(input = 'V.txt', ID_me=1, ID_otherguy=2):
    f = open(input)

    text = f.read()

    text = ''.join(text.split()) # Remove whitespace
    text = text.replace('^','**') # Replace ^ with **
    
    text = text.replace('a0','p' + str(ID_me) + '[0]')
    text = text.replace('a1','p' + str(ID_me) + '[1]')
    text = text.replace('a2','p' + str(ID_me) + '[2]')
    text = text.replace('a3','p' + str(ID_me) + '[3]')
    text = text.replace('a4','p' + str(ID_me) + '[4]')
    
    text = text.replace('b0','p' + str(ID_otherguy) + '[0]')
    text = text.replace('b1','p' + str(ID_otherguy) + '[1]')
    text = text.replace('b2','p' + str(ID_otherguy) + '[2]')
    text = text.replace('b3','p' + str(ID_otherguy) + '[3]')
    text = text.replace('b4','p' + str(ID_otherguy) + '[4]')

    num_terms = text.count('{')

    # Extract terms and get rid of curly braces
    terms = []
    while text.find('}') > -1:
        cut_point = text.find('}')
        terms.append(text[:cut_point].strip('{'))
        text = text[cut_point+1:]
        
    return terms