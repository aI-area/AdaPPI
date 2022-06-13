def precision_score(predicted_complex, reference_complex, predicted_num):
    match_info_list = []
    unmatch_pred_list = []
    number=0
    for i, pred in enumerate(predicted_complex):
        overlapscore=0.0
        tmp_max_score_info = None
        for j, ref in enumerate(reference_complex):
            set1 = set(ref)
            set2 = set(pred)
            overlap = set1 & set2
            score = float((pow(len(overlap), 2))) / float((len(set1) * len(set2)))
            if score > overlapscore:
                overlapscore = score
                # find max score
                tmp_max_score_info = {
                    'pred': pred, 'true': ref, 
                    'overlap_score': overlapscore,
                    'pred_id': i, 'true_id': j, 
                }
        if overlapscore > 0.25:
            number = number + 1
            if tmp_max_score_info is not None:
                match_info_list.append(tmp_max_score_info)                
        else:
            unmatch_pred_list.append(pred)

    return number/(predicted_num+1e-4), number, match_info_list, unmatch_pred_list


def recall_score(predicted_complex, reference_complex, reference_num):
    match_info_list = []
    unmatch_pred_list = []
    c_number = 0
    for i, ref in enumerate(reference_complex):
        overlapscore=0.0
        tmp_max_score_info = None
        for j, pred in enumerate(predicted_complex):
            set1 = set(ref)
            set2 = set(pred)
            overlap = set1 & set2
            score = float((pow(len(overlap), 2))) / float((len(set1) * len(set2)))
            
            if score > overlapscore:
                overlapscore = score

                tmp_max_score_info = {
                    'pred': pred, 'true': ref, 
                    'overlap_score': overlapscore,
                    'pred_id': j, 'true_id': i, 
                }
            
        if overlapscore > 0.25:
            c_number = c_number+1
            if tmp_max_score_info is not None:
                match_info_list.append(tmp_max_score_info)
        else:
            unmatch_pred_list.append(pred)

    return c_number/(reference_num+1e-4), c_number, match_info_list, unmatch_pred_list


def acc_score(predicted_complex, reference_complex):
    # sn
    T_sum1=0.0
    N_sum=0.0   # the number of proteins in reference complex
    for i in reference_complex:
        max=0.0
        for j in predicted_complex:
            set1=set(i)
            set2=set(j)
            overlap=set1&set2
            if len(overlap)>max:
                max=len(overlap)
        T_sum1=T_sum1+max
        N_sum=N_sum+len(set1)

    # ppv
    T_sum2=0.0
    T_sum=0.0
    for i in predicted_complex:
        max=0.0
        for j in reference_complex:
            set1=set(i)
            set2=set(j)
            overlap=set1&set2
            T_sum=T_sum+len(overlap)
            if len(overlap)>max:
                max=len(overlap)
        T_sum2=T_sum2+max
    
    Sn = float(T_sum1) / float(N_sum)
    PPV = float(T_sum2) / float(T_sum+1e-5)
    Acc = pow(float(Sn*PPV), 0.5)
    return Acc


def get_score(reference_complex, predicted_complex):

    acc = acc_score(predicted_complex, reference_complex)

    reference_num = len(reference_complex)
    predicted_num = len(predicted_complex)

    precision, p_num, _, _ = precision_score(predicted_complex, reference_complex, predicted_num)
    recall, r_num, _, _ = recall_score(predicted_complex, reference_complex, reference_num)

    f1 = float((2*precision*recall)/(precision+recall+1e-5))

    msg = "[precision={:.4f}, recall={:.4f}, f1={:.4f}, acc={:.4f}]".format(precision, recall, f1, acc)

    num_msg = f' precision={p_num}/{predicted_num}, recall={r_num}/{reference_num} '

    return msg+num_msg
