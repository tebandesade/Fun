import calendar as cal

class ViajeEsteban:
    def calculateDays(initialDay, initialMonth, initialYear,targetDay, targetMonth, targetYear):
        #Se incluye el dia inicial
        diasIniciales = cal.monthrange(initialYear, initialMonth)[1] - initialDay+1;
        #print "Dias iniciales" , diasIniciales;
        #Se incluye el dia final
        diasFinales = targetDay ;
        #print "Dias finales" ,diasFinales;
        totalDays= diasIniciales + diasFinales;
        #Si es en el mismo anio
        if initialYear == targetYear:
            veces = targetMonth - initialMonth -1;
            #print "veces son:",veces;
            for x in range(veces):
                initialMonth += initialMonth;
                totalDays = totalDays + cal.monthrange(initialYear,initialMonth )[1];
            return totalDays;
        #Si no es en el mismo anio
        elif initialYear < targetYear:
            difYears = targetYear - initialYear;
            veces = targetMonth-initialMonth + difYears * 12 -1;
            for x in range(veces):
                initialMonth += initialMonth;
                if(initialMonth>12):
                    initialYear += initialYear;
                    initialMonth = 01;
                    totalDays = totalDays + cal.monthrange(initialYear, initialMonth)[1];
                totalDays = totalDays + cal.monthrange(initialYear, initialMonth)[1];
            return totalDays;

    diasViaje = calculateDays(20,01,2017,28,05,2017);
    totalSemanas = diasViaje/7;
    precioMetro = 33;

    print "Total metro card:", precioMetro * totalSemanas;
    print "Total food:", diasViaje * 30;
