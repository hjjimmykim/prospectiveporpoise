def mathematica2py(input = 'V.txt'):
    f = open(input)

    text = f.read()

    text = ''.join(text.split()) # Remove whitespace
    text = text.replace('^','**') # Replace ^ with **

    num_terms = text.count('{')

    # Extract terms and get rid of curly braces
    terms = []
    while text.find('}') > -1:
        cut_point = text.find('}')
        terms.append(text[:cut_point].strip('{'))
        text = text[cut_point+1:]
        
    return terms