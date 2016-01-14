    output_file=path+'output'+str(t)+'.text'
    f = open(output_file, 'w')
    for node in state.keys():
        np.savez(f, node, state[node])
    f.seek(0) # Only needed here to simulate closing & reopening file
    npzfile = np.load(f)
    npzfile[node]
    np.savetxt(f, B)