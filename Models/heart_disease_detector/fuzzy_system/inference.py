class FuzzyInference:
    def __int__(self):
        pass

    def inference_result(self, age, bp, bs, cholesterol, hr, ecg, op, cp, exercise, thallium, sex):
        output_sick, output_sick, output_sick, output_sick, output_healthy = ([] for i in range(5))
        '''
        This function take as input the dict containing all input attributes, 
        and gives the fuzzy output based on classification_rules.
        '''

        # rules
        # 1
        output_sick.append(min(float(age['very_old']), float(cp['atypical_anginal'])))
        # 2
        output_sick.append(min(float(hr['high']), float(age['old'])))
        # 3
        output_sick.append(min(float(sex['male']), float(hr['medium'])))
        # 4
        output_sick.append(min(float(sex['female']), float(hr['medium'])))
        # 5
        output_sick.append(min(float(cp['non_anginal_pain']), float(bp['high'])))
        # 6
        output_sick.append(min(float(cp['typical_anginal']), float(hr['medium'])))
        # 7
        output_sick.append(min(float(bs['true']), float(age['mid'])))
        # 8
        output_sick.append(min(float(bs['false']), float(bp['very_high'])))
        # 9
        output_sick.append(max(float(cp['asymptomatic']), float(age['very_old'])))
        # 10
        output_sick.append(max(float(bp['high']), float(age['very_old'])))

        # Chest Pain
        # 11
        output_healthy.append(float(cp['typical_anginal']))
        # 12
        output_sick.append(float(cp['atypical_anginal']))
        # 13
        output_sick.append(float(cp['non_anginal_pain']))
        # 14
        output_sick.append(float(cp['asymptomatic']))

        # Sex
        # 16
        output_healthy.append(float(sex['female']))
        # 17
        output_healthy.append(float(sex['male']))

        # Blood Pressure
        # 18
        output_healthy.append(float(bp['low']))
        # 19
        output_sick.append(float(bp['medium']))
        # 20
        output_sick.append(float(bp['high']))
        # 21
        output_sick.append(float(bp['very_high']))

        # Cholesterol
        # 22
        output_healthy.append(float(cholesterol['low']))
        # 23
        output_sick.append(float(cholesterol['medium']))
        # 24
        output_sick.append(float(cholesterol['high']))
        # 25
        output_sick.append(float(cholesterol['very_high']))

        # Blood sugar
        # 26
        output_sick.append(float(bs['true']))
        # 27
        output_healthy.append(float(bs['false']))

        # ECG
        # 28
        output_healthy.append(float(ecg['normal']))
        # 29
        output_sick.append(float(ecg['abnormal']))
        # 30
        output_sick.append(float(ecg['hypertrophy']))

        # Heart Rate
        # 31
        output_healthy.append(float(hr['low']))
        # 32
        output_sick.append(float(hr['medium']))
        # 33
        output_sick.append(float(hr['high']))

        # exercise
        # 34
        output_sick.append(float(exercise['true']))
        # 35
        output_healthy.append(float(exercise['false']))

        # Old Peak
        # 36
        output_healthy.append(float(op['low']))
        # 37
        output_sick.append(float(op['terrible']))
        # 38
        output_sick.append(float(op['risk']))

        # Thallium
        # 39
        output_healthy.append(float(thallium['normal']))
        # 40
        output_sick.append(float(thallium['medium']))
        # 41
        output_sick.append(float(thallium['high']))

        # Age
        # 42
        output_healthy.append(float(age['young']))
        # 43
        output_sick.append(float(age['mid']))
        # 44
        output_sick.append(float(age['old']))
        # 45
        output_sick.append(float(age['very_old']))

        return dict(
            output_sick=max(output_sick),
            output_healthy=max(output_healthy)
        )
