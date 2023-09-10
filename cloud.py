import numpy as np


def gaussian_pdf(x, mean, std):
    exponent = np.exp(-((x-mean)**2 / (2 * std**2 )))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

def classify(sample_embedding, class_means, class_std_devs):
    likelihoods = {}
    for class_name in class_means.keys():
        likelihood = np.prod(gaussian_pdf(sample_embedding, class_means[class_name], class_std_devs[class_name]))
        likelihoods[class_name] = likelihood
        
    # Return the class with the maximum likelihood
    return max(likelihoods, key=likelihoods.get)



sample = np.array([...]) # Your 44-dimensional sample embedding
predicted_class = classify(sample, class_means, class_std_devs)
print(predicted_class)
