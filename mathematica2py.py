def mathematica2py(input = 'V.txt'):
    f = open(input)

    text = f.read()

    text = ''.join(text.split()) # Remove whitespace
    text = text.replace('^','**') # Replace ^ with **
    
    text = text.replace('a0','p1[0]')
    text = text.replace('a1','p1[1]')
    text = text.replace('a2','p1[2]')
    text = text.replace('a3','p1[3]')
    text = text.replace('a4','p1[4]')
    
    text = text.replace('b0','p2[0]')
    text = text.replace('b1','p2[1]')
    text = text.replace('b2','p2[2]')
    text = text.replace('b3','p2[3]')
    text = text.replace('b4','p2[4]')

    num_terms = text.count('{')

    # Extract terms and get rid of curly braces
    terms = []
    while text.find('}') > -1:
        cut_point = text.find('}')
        terms.append(text[:cut_point].strip('{'))
        text = text[cut_point+1:]
        
    return terms