from Models.heart_disease_detector.fuzzy_system.fuzzification import OutPutSick
output_sick = OutPutSick()


class Defuzzify:
    def __int__(self):
        pass

    def defuzzy_caculator(self, data):
        points_num = 1000
        step = 5. / points_num
        points_of_sickness = [0 + i * step for i in range(points_num + 1)]

        numerator = 0.
        denominator = 0.

        for point in points_of_sickness:

            sick = output_sick.outPut_sick(point)
            sick = data['output_sick'] if sick > data['output_sick'] else sick

            healthy = output_sick.healthy(point)
            healthy = data['output_healthy'] if healthy > data['output_healthy'] else healthy

            result = max(sick, healthy)

            numerator += result * point
            denominator += result

        return numerator/denominator if denominator != 0 else 0

    def defuzzification_result(self, data):
        x_star = self.defuzzy_caculator(data)
        result = ""
        if x_star < 1.78:
             result += "healthy "
        else:
            result += "sick"
        result += ": " + str(x_star)
        return result
