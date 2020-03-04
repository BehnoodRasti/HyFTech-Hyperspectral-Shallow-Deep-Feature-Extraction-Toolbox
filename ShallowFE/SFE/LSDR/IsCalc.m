function Is_new = IsCalc(Hhat,hhat,alpha)
    Is_new = alpha'*Hhat*alpha/2 - hhat'*alpha;