
#Dados rio East Fork (DADOS DE VALOR INICIAL E CONTORNO)
#Largura do rio em função a distância em metros
function anchofork(x)
    #  Na entrada x pode ter qualquer valor mesmo negativo o maior que 3213
    #  A tabela vem dada para x entre 0 e 3213
    #  Se x está fora desse intervalo, esta subrotina calcula a
    #  largura do canal  por extrapolacao.

    #  Largura do Rio Fork  entre x = 0  e  x = 3213.
    #  (Na metrica de Ayvaz entre x = 3256 e  x = 43)
    #  (x = 3256  en Ayvaz corresponde a x = 0 neste programa)


    metros = [ 
        0.00000E+00,    0.88000E+02,    0.14800E+03,    0.20900E+03,
        0.29500E+03,    0.38200E+03,    0.47800E+03,    0.56600E+03,
        0.64800E+03,    0.74600E+03,    0.83400E+03,    0.90000E+03,
        0.97800E+03,    0.10620E+04,    0.11740E+04,    0.12600E+04,
        0.13550E+04,    0.14260E+04,    0.14900E+04,    0.15610E+04,
        0.15940E+04,    0.17750E+04,    0.18600E+04,    0.19410E+04,
        0.20150E+04,    0.21010E+04,    0.21790E+04,    0.22710E+04,
        0.23580E+04,    0.24480E+04,    0.25480E+04,    0.26540E+04,
        0.27400E+04,    0.28350E+04,    0.29550E+04,    0.30360E+04,
        0.31190E+04,    0.31810E+04,    0.32130E+04]

    ancho = [
        0.16500E+02,    0.11500E+02,    0.14500E+02,    0.17500E+02,
        0.15500E+02,    0.25500E+02,    0.21000E+02,    0.15500E+02,
        0.20500E+02,    0.19000E+02,    0.20500E+02,    0.20000E+02,
        0.24000E+02,    0.15500E+02,    0.20000E+02,    0.15000E+02,
        0.14500E+02,    0.22500E+02,    0.19000E+02,    0.17000E+02,
        0.16500E+02,    0.26000E+02,    0.12000E+02,    0.22500E+02,
        0.18000E+02,    0.22000E+02,    0.22000E+02,    0.30500E+02,
        0.14000E+02,    0.18500E+02,    0.16000E+02,    0.19500E+02,
        0.14500E+02,    0.23000E+02,    0.14000E+02,    0.16000E+02,
        0.14500E+02,    0.15000E+02,    0.16000E+02]

    faux = interpolate((metros,), ancho, Gridded(Linear())) # criando uma interpolação no dom [1,naux]
    faux = extrapolate(faux, Line()) # extrapolação linear nas pontas

    return faux(x)
end

###### Dados de elevação z em função da distância em metros ########
function zfork(x)
    #  Na entrada x pode ter qualquer valor mesmo negativo o maior que 3213
    #  A tabela vem dada para x entre 0 e 3213
    #  Se x está fora desse intervalo, esta subrotina calcula a elevacao z por extrapolacao.

    #  Condicao inicial na elevacao z entre x = 0  e  x = 3213.
    #  (Na metrica de Ayvaz entre x = 3256 e  x = 43 )
    #  (x = 3256  en Ayvaz corresponde a x = 0 neste programa)



    metros = [ 
        0.00000E+00,    0.88000E+02,    0.14800E+03,    0.20900E+03,
        0.29500E+03,    0.38200E+03,    0.47800E+03,    0.56600E+03,
        0.64800E+03,    0.74600E+03,    0.83400E+03,    0.90000E+03,
        0.97800E+03,    0.10620E+04,    0.11740E+04,    0.12600E+04,
        0.13550E+04,    0.14260E+04,    0.14900E+04,    0.15610E+04,
        0.15940E+04,    0.17750E+04,    0.18600E+04,    0.19410E+04,
        0.20150E+04,    0.21010E+04,    0.21790E+04,    0.22710E+04,
        0.23580E+04,    0.24480E+04,    0.25480E+04,    0.26540E+04,
        0.27400E+04,    0.28350E+04,    0.29550E+04,    0.30360E+04,
        0.31190E+04,    0.31810E+04,    0.32130E+04]

    z = [
        0.82500E+01,    0.81700E+01,    0.81700E+01,    0.81400E+01,
        0.81000E+01,    0.80700E+01,    0.80400E+01,    0.79700E+01,
        0.79200E+01,    0.79000E+01,    0.78000E+01,    0.77900E+01,
        0.77300E+01,    0.76200E+01,    0.75800E+01,    0.75000E+01,
        0.74400E+01,    0.74000E+01,    0.73900E+01,    0.73100E+01,
        0.73100E+01,    0.71400E+01,    0.70200E+01,    0.69600E+01,
        0.70400E+01,    0.69900E+01,    0.69100E+01,    0.68700E+01,
        0.67400E+01,    0.66600E+01,    0.66000E+01,    0.65400E+01,
        0.64800E+01,    0.64300E+01,    0.63500E+01,    0.62900E+01,
        0.62400E+01,    0.62200E+01,    0.61700E+01]

    faux = interpolate((metros,), z, Gridded(Linear())) # criando uma interpolação no dom [1,naux]
    faux = extrapolate(faux, Line()) # extrapolação linear nas pontas

    return faux(x)

end

###### Dados de cota de fundo zb em função da distância em metros ########
function zbfork(x)
    #  Na entrada x pode ter qualquer valor mesmo negativo o maior que 3213
    #  A tabela vem dada para x entre 0 e 3213
    #  Se x está fora desse intervalo, esta subrotina calcula cotafun por extrapolacao.

    #  Cota de fundo z_b entre x = 0  e  x = 3213.
    #  (Na metrica de Ayvaz entre x = 3256 e  x = 43)
    #  (x = 3256  en Ayvaz corresponde a x = 0 neste programa)

    metros = [ 
        0.00000E+00,    0.88000E+02,    0.14800E+03,    0.20900E+03,
        0.29500E+03,    0.38200E+03,    0.47800E+03,    0.56600E+03,
        0.64800E+03,    0.74600E+03,    0.83400E+03,    0.90000E+03,
        0.97800E+03,    0.10620E+04,    0.11740E+04,    0.12600E+04,
        0.13550E+04,    0.14260E+04,    0.14900E+04,    0.15610E+04,
        0.15940E+04,    0.17750E+04,    0.18600E+04,    0.19410E+04,
        0.20150E+04,    0.21010E+04,    0.21790E+04,    0.22710E+04,
        0.23580E+04,    0.24480E+04,    0.25480E+04,    0.26540E+04,
        0.27400E+04,    0.28350E+04,    0.29550E+04,    0.30360E+04,
        0.31190E+04,    0.31810E+04,    0.32130E+04]

    zb = [
        0.74600E+01,    0.69100E+01,    0.71300E+01,    0.72100E+01,
        0.71400E+01,    0.74200E+01,    0.70400E+01,    0.68100E+01,
        0.70100E+01,    0.72100E+01,    0.65600E+01,    0.70200E+01,
        0.68300E+01,    0.66100E+01,    0.68000E+01,    0.64100E+01,
        0.65600E+01,    0.66200E+01,    0.65000E+01,    0.64800E+01,
        0.62500E+01,    0.62100E+01,    0.62000E+01,    0.61400E+01,
        0.62500E+01,    0.63800E+01,    0.59900E+01,    0.62900E+01,
        0.58500E+01,    0.58700E+01,    0.55700E+01,    0.54800E+01,
        0.53900E+01,    0.56400E+01,    0.52700E+01,    0.53800E+01,
        0.51500E+01,    0.49400E+01,    0.49400E+01]

      
    faux = interpolate((metros,), zb, Gridded(Linear())) # criando uma interpolação no dom [1,naux]
    faux = extrapolate(faux, Line()) # extrapolação linear nas pontas

    return faux(x)
end

###### Dados da elevação em 751 em função do tempo em segundos ########
function zhistomedio(t)
    #  Parameters:
    #  t: time (in seconds) at which you want to compute z(751, t).
    #  z: value of z(751, t) computed by this subroutine.

    #  En la entrada,
    #  t es tiempo en segundos, contado desde el 17 de mayo.
    #  O sea, en la entrada de esta subrutina, la variable t
    #  significa cantidad de segundos transcurridos a partir
    #  del 17 de mayo.



    tempo = [ 
        72.0,   84.0, 96.0,  108.0,
        120.0, 132.0, 144.0, 156.0,
        168.0, 180.0, 192.0, 204.0,
        216.0, 228.0, 240.0, 252.0,
        264.0, 276.0, 288.0, 300.0,
        312.0, 324.0, 336.0, 348.0,
        360.0, 372.0, 384.0, 396.0,
        408.0, 420.0, 432.0, 444.0,
        456.0, 468.0, 480.0, 492.0,
        504.0, 516.0, 528.0, 540.0,
        552.0, 564.0, 576.0, 588.0,
        600.0, 612.0, 624.0, 636.0,
        648.0, 660.0, 672.0, 684.0,
        696.0, 708.0, 720.0, 732.0,
        744.0, 756.0, 768.0, 780.0]

    elev = [
        0.77600E+01,    0.78600E+01,    0.79300E+01,    0.79600E+01,
        0.80300E+01,    0.81500E+01,    0.81000E+01,    0.82100E+01,
        0.81400E+01,    0.82600E+01,    0.81900E+01,    0.82500E+01,
        0.82000E+01,    0.82700E+01,    0.82500E+01,    0.83800E+01,
        0.83100E+01,    0.83300E+01,    0.82600E+01,    0.83000E+01,
        0.82900E+01,    0.80900E+01,    0.79100E+01,    0.77100E+01,
        0.76100E+01,    0.74600E+01,    0.74300E+01,    0.73800E+01,
        0.73900E+01,    0.75200E+01,    0.75300E+01,    0.77700E+01,
        0.77700E+01,    0.80000E+01,    0.79300E+01,    0.81200E+01,
        0.80100E+01,    0.79200E+01,    0.77800E+01,    0.76200E+01,
        0.75200E+01,    0.73700E+01,    0.73600E+01,    0.73600E+01,
        0.73500E+01,    0.75800E+01,    0.75500E+01,    0.77400E+01,
        0.76800E+01,    0.80000E+01,    0.78400E+01,    0.81100E+01,
        0.79100E+01,    0.80300E+01,    0.77500E+01,    0.77300E+01,
        0.75900E+01,    0.76300E+01,    0.75100E+01,    0.75800E+01]

	
	horas = t/3600.0

    faux = interpolate((tempo,), elev, Gridded(Linear())) # criando uma interpolação no dom [1,naux]
    faux = extrapolate(faux, Line()) # extrapolação linear nas pontas

    return faux(horas)
	
end

###### Dados da elevação em 3256 em função do tempo em segundos ########
function zoutlet(t)
    #  Na entrada, t es tiempo en segundos, contado desde el 17 de mayo.
    #  O sea, en la entrada de esta subrutina, la variable t
    #  significa cantidad de segundos transcurridos a partir
    #  del 17 de mayo.

    #  Parameters:
    #  t: time (seconds) at which you want z(3256, t)
    #  elev: value of z(3256, t) computed by this subroutine

    #  Hidrograma de z em x = 0000 na metrica de Ayvaz, o qual
    #  corresponde a x = 3213 + 43 = 3256 na metrica deste programa.
    #  (x = 3256  en Ayvaz corresponde a x = 0 neste programa)

    #  Na seguinte tabela times=0 corresponde ao 17 de maio
    #  times=72 corresponde a 20 de maio
        
    tempo = [ 
        72.0,   84.0, 96.0,  108.0,
        120.0, 132.0, 144.0, 156.0,
        168.0, 180.0, 192.0, 204.0,
        216.0, 228.0, 240.0, 252.0,
        264.0, 276.0, 288.0, 300.0,
        312.0, 324.0, 336.0, 348.0,
        360.0, 372.0, 384.0, 396.0,
        408.0, 420.0, 432.0, 444.0,
        456.0, 468.0, 480.0, 492.0,
        504.0, 516.0, 528.0, 540.0,
        552.0, 564.0, 576.0, 588.0,
        600.0, 612.0, 624.0, 636.0,
        648.0, 660.0, 672.0, 684.0,
        696.0, 708.0, 720.0, 732.0,
        744.0, 756.0, 768.0, 780.0,
        792.0, 804.0]

    elev = [
        0.58200E+01,    0.59700E+01,    0.59900E+01,    0.60900E+01,
        0.60900E+01,    0.62800E+01,    0.61800E+01,    0.63600E+01,
        0.62300E+01,    0.64300E+01,    0.63000E+01,    0.63900E+01,
        0.63100E+01,    0.64100E+01,    0.63700E+01,    0.66100E+01,
        0.64800E+01,    0.65300E+01,    0.63900E+01,    0.64600E+01,
        0.64500E+01,    0.61700E+01,    0.60000E+01,    0.58200E+01,
        0.57300E+01,    0.56300E+01,    0.55800E+01,    0.55500E+01,
        0.55200E+01,    0.56400E+01,    0.56500E+01,    0.58400E+01,
        0.58400E+01,    0.60700E+01,    0.59700E+01,    0.61800E+01,
        0.60800E+01,    0.60200E+01,    0.58700E+01,    0.57200E+01,
        0.56400E+01,    0.55200E+01,    0.55100E+01,    0.55100E+01,
        0.55000E+01,    0.56900E+01,    0.56700E+01,    0.58300E+01,
        0.57700E+01,    0.60600E+01,    0.59100E+01,    0.61800E+01,
        0.60000E+01,    0.61100E+01,    0.58500E+01,    0.58300E+01,
        0.56900E+01,    0.57400E+01,    0.56000E+01,    0.56900E+01,
        0.56000E+01,    0.55300E+01]

        
	horas = t/3600.0

    faux = interpolate((tempo,), elev, Gridded(Linear())) # criando uma interpolação no dom [1,naux]
    faux = extrapolate(faux, Line()) # extrapolação linear nas pontas

    return faux(horas)
    
	
end
#####################################################################

function qinlet(t)
    #  t es tiempo en segundos, contado desde el 17 de mayo.
    #  O sea, en la entrada de esta subrutina, la variable t
    #  significa cantidad de segundos transcurridos a partir
    #  del 17 de mayo.

    #  Parameters:
    #  t: time (seconds) at which you want the value of q(-39, t).

    #  Hidrograma de Q em x = 3295  na metrica de Ayvaz, o qual
    #  corresponde a x = -39  na metrica deste programa.
    #  (x = 3295  en Ayvaz corresponde a x = -39 neste programa)
    #  Tiempo cero corresponde al 17 de mayo.

    #  Given t in seconds this subroutine computes the value of Q(x=0, t)

    #  Datos con tiempo cero = 0   y    tiempo final = 46.5 dias
        


    tempo = [ 
        0.00000E+00,    0.12000E+02,    0.24000E+02,    0.36000E+02,
        0.48000E+02,    0.60000E+02,    0.72000E+02,    0.84000E+02,
        0.96000E+02,    0.10800E+03,    0.12000E+03,    0.13200E+03,
        0.14400E+03,    0.15600E+03,    0.16800E+03,    0.18000E+03,
        0.19200E+03,    0.20400E+03,    0.21600E+03,    0.22800E+03,
        0.24000E+03,    0.25200E+03,    0.26400E+03,    0.27600E+03,
        0.28800E+03,    0.30000E+03,    0.31200E+03,    0.32400E+03,
        0.33600E+03,    0.34800E+03,    0.36000E+03,    0.37200E+03,
        0.38400E+03,    0.39600E+03,    0.40800E+03,    0.42000E+03,
        0.43200E+03,    0.44400E+03,    0.45600E+03,    0.46800E+03,
        0.48000E+03,    0.49200E+03,    0.50400E+03,    0.51600E+03,
        0.52800E+03,    0.54000E+03,    0.55200E+03,    0.56400E+03,
        0.57600E+03,    0.58800E+03,    0.60000E+03,    0.61200E+03,
        0.62400E+03,    0.63600E+03,    0.64800E+03,    0.66000E+03,
        0.67200E+03,    0.68400E+03,    0.69600E+03,    0.70800E+03,
        0.72000E+03,    0.73200E+03,    0.74400E+03,    0.75600E+03,
        0.76800E+03,    0.78000E+03,    0.79200E+03,    0.80400E+03,
        0.81600E+03,    0.82800E+03,    0.84000E+03,    0.85200E+03,
        0.86400E+03,    0.87600E+03,    0.88800E+03,    0.90000E+03,
        0.91200E+03,    0.92400E+03,    0.93600E+03,    0.94800E+03,
        0.96000E+03,    0.97200E+03,    0.98400E+03,    0.99600E+03,
        0.10080E+04,    0.10200E+04,    0.10320E+04,    0.10440E+04,
        0.10560E+04,    0.10680E+04,    0.10800E+04,    0.10920E+04,
        0.11040E+04,    0.11160E+04]

    conto = [
        0.39000E+01,    0.39000E+01,    0.49800E+01,    0.57400E+01,
        0.72000E+01,    0.67800E+01,    0.87600E+01,    0.10700E+02,
        0.12400E+02,    0.13000E+02,    0.14500E+02,    0.17500E+02,
        0.16700E+02,    0.20900E+02,    0.18400E+02,    0.25200E+02,
        0.20900E+02,    0.24700E+02,    0.22300E+02,    0.25500E+02,
        0.24900E+02,    0.33100E+02,    0.28200E+02,    0.30000E+02,
        0.25500E+02,    0.29100E+02,    0.27400E+02,    0.16100E+02,
        0.12700E+02,    0.94000E+01,    0.76200E+01,    0.53500E+01,
        0.49800E+01,    0.41800E+01,    0.41800E+01,    0.55800E+01,
        0.66200E+01,    0.96800E+01,    0.10500E+02,    0.14600E+02,
        0.13400E+02,    0.19000E+02,    0.15100E+02,    0.13600E+02,
        0.10600E+02,    0.76200E+01,    0.59800E+01,    0.41100E+01,
        0.37600E+01,    0.32300E+01,    0.36900E+01,    0.67000E+01,
        0.65400E+01,    0.94000E+01,    0.87600E+01,    0.15200E+02,
        0.11800E+02,    0.19000E+02,    0.12800E+02,    0.15400E+02,
        0.98500E+01,    0.94000E+01,    0.71200E+01,    0.74500E+01,
        0.52000E+01,    0.62900E+01,    0.48300E+01,    0.36900E+01,
        0.30400E+01,    0.25400E+01,    0.20200E+01,    0.19700E+01,
        0.16500E+01,    0.26700E+01,    0.20200E+01,    0.27300E+01,
        0.20800E+01,    0.27900E+01,    0.20800E+01,    0.29800E+01,
        0.22500E+01,    0.26700E+01,    0.21900E+01,    0.32300E+01,
        0.21900E+01,    0.23700E+01,    0.19100E+01,    0.24300E+01,
        0.18600E+01,    0.23700E+01,    0.19100E+01,    0.17000E+01,
        0.16500E+01,    0.13900E+01]

	horas = t/3600.0

    faux = interpolate((tempo,), conto, Gridded(Linear())) # criando uma interpolação no dom [1,naux]
    faux = extrapolate(faux, Line()) # extrapolação linear nas pontas

    return faux(horas)
	
end