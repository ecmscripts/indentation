

def train_network(net, data, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    # Use CrossEntropyLoss instead of NLLLoss
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    total_loss = 0.0
    accuracy   = 0

    for epoch in range(num_epochs):
        clear_output(wait=True)
        net.train()
        running_loss = 0.0

        print("Total Progress:\t\t", np.round((epoch+1)/num_epochs*100, 2), "%")
        print(f'Accuracy: \t\t {accuracy:.2f}%\n')
        
        for batch_idx, (X, y) in enumerate(data):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            output = net(X)
            
            loss = F.nll_loss(output, y)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print batch loss every 10 batches
            if batch_idx % 10 == 9:
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss/10:.4f}')
                running_loss = 0.0
        
        # Evaluate on the entire dataset after each epoch
        net.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for X, y in data:
                X, y = X.to(device), y.to(device)
                outputs = net(X)
                total_loss += F.nll_loss(output, y).item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        avg_loss = total_loss / len(data)
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')


def create_data_sets(displ_max = 100, step_size = 0.05, alpha = 0.1, alpha_var = 0.01, 
                     noise_scale1 = 2, noise_scale2 = 0.1, noise_var=0.5, baseline_tilt_max=0.1,
                     batch_size = 32, N_batches  = 50, N=100):

    N_gen_data = N_batches*batch_size

    data_all_tests     = []
    data_all_tests_img = []
    for i in range(N_gen_data):

        # base indentation to be modded
        displ_base, force_base = indentation_force(np.linspace(0, displ_max, int(displ_max/step_size)), displ_max, alpha + alpha_var*np.random.rand())

        # random shift data to obtain new reference point
        xs = displ_max/3 + 0.7*displ_max*np.random.rand()
        
        # add noise to indentation data
        ys = 3*np.random.rand()
        displ, force = shift(displ_base, force_base, displ_max, xs, ys)
        force        = add_noise(force, noise_scale1, noise_scale2, noise_var)
        force        = add_baseline_tilt(force, baseline_tilt_max)

        # clip data
        displ, force = clip_data(displ, force)
        xs_norm      = xs/np.max(displ)
        displ, force = normalize(displ, force, N)
        
        # calculate shift index
        ix_shift = np.argmin(np.abs(np.linspace(0,1,N)-xs_norm))

        # compute image
        force_img = create_convolution_image(force, N=N)
                            
        # append data signal-ground_truth pair 
        data_all_tests.append([torch.tensor(force, dtype=torch.float).view(-1 ,N), int(ix_shift)])
        data_all_tests_img.append([torch.tensor(force_img, dtype=torch.float).view(-1 ,N, N), int(ix_shift)])

    # create batches
    data     = []
    data_img = []

    for i in range(N_batches):
        datax = []
        datay = []
        dataximg = []
        datayimg = []
        for j in range(batch_size):
            ix = i*batch_size+j
            X, y = data_all_tests[ix]
            Ximg, yimg = data_all_tests_img[ix]
            datax.append(X)
            datay.append(y)
            dataximg.append(Ximg)
            datayimg.append(yimg)
        data.append([torch.stack(datax), torch.tensor(datay)])
        data_img.append([torch.stack(dataximg), torch.tensor(datayimg)])
    return data, data_img
