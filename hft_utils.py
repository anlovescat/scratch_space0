def print_book(book, precition=4):
    string = 'Date: %s\n'%(book['date'].strftime('%Y-%m-%d'))
    string+= 'Time: %s\n'%(book['time'].strftime('%H:%M:%S'))
    string+= 'Volume: %d\n' %(book['volume'])
    string+= 'CumVolume: %d\n' %(book['cumvolume'])
    if 'oichange' in book.dtype.names:
        string+= 'OIChange: %d\n' %(book['oichange'])
    elif 'volumevalue' in book.dtype.names :
        string+= 'VolumeValue: %.2f\n' %(book['volumevalue'])
        string+= 'VWAP: %.5f\n' %(book['volumevalue'] / book['volume'] / 100.0)
    string+= '==================================================\n'
    string+= 'Level     Price     Size      LastDir   LastPrice \n'
    price_format = '%.'+str(precition)+'f' + '   '
    for ii in range(3, 0, -1):
        ask = 'ask' + str(ii)
        asksize = 'asksize' + str(ii)
        ask_str = 'Ask_' + str(ii)
        string += ask_str + ' '*5 + (price_format%(book[ask])).rjust(10) + ('%3d    '%(book[asksize])).rjust(10)
        string += '\n'
    if book['lastdirection'] == 'B':
        string += ' '*30 + '%s         '%(book['lastdirection']) + (price_format%(book['lastprice'])).rjust(10) + '\n'
        string += '-'*50 + '\n'
        string += '\n'
    else :
        string += '\n'
        string += '-'*50 + '\n'
        string += ' '*30 + '%s         '%(book['lastdirection']) + (price_format%(book['lastprice'])).rjust(10) + '\n'
    for ii in range(1, 4, 1):
        bid = 'bid' + str(ii)
        bidsize = 'bidsize' + str(ii)
        bid_str = 'Bid_' + str(ii)
        string += bid_str + ' '*5 + (price_format%(book[bid])).rjust(10) + ('%3d    '%(book[bidsize])).rjust(10)
        string += '\n'
        
    print string


def print_book_history(data, precision=3):
    print "press Enter to continue; other key to exit"
    ii = 0
    while(raw_input() == '') :
        print_book(data[ii], precision)
        ii += 1
    return 0


