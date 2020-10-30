! This is a simple example program to illustrate the use of
! OclWrapperF, my OpenCL wrapper for Fortran
! (c) Wim Vanderbauwhede 2011-2012

      program MatrixAccumulation
        use oclWrapper
        implicit none
        ! Variables set via cpp macros
        integer :: nruns    
        integer :: mSize
        integer :: mWidth 
        integer :: knum
        ! loop counters
        integer :: i,j,k,run

        real:: mCref
        real, allocatable, dimension(:)  :: mA 
        real, allocatable, dimension(:)  :: mC 

        integer :: nunits
        character(15) :: kstr ! TODO: dynamic allocation
        character(10) :: srcstr  ! TODO: dynamic allocation

        !integer(8):: ocl
        integer(8):: mA_buf
        integer(8):: mC_buf

#ifdef VERBOSE        
        ! for comparison with reference        
        real::mCtot
        integer :: correct
#endif        
        ! for timings
        real, dimension(2) :: tarray
        real :: result

        knum= KERNEL
        nruns=NRUNS
        mSize = WIDTH*WIDTH
        mWidth = WIDTH       
        print *, mWidth
        allocate( mA(mSize) )
        ! Create the data sets   
        mA = (/ (k*0.0+1.0/mSize, k=1,mSize) /)
#if REF!=0
        mCref=0.0
        do run=1,nruns
            mCref=0.0
            call dtime(tarray, result)
            do i = 1,mWidth 
                do j = 1,mWidth 
                    mCref= mCref + mA((i-1)*mWidth+j)
                end do
            end do
            call dtime(tarray, result)
#ifdef VERBOSE
        print *,  "Execution time for reference: ",result*1000," ms"
#else
        print *,  " ",result*1000, " "
#endif
        end do
#endif
#if REF!=2
        !--------------------------------------------------------------------------------
        !---- Here starts the actual OpenCL part
        !--------------------------------------------------------------------------------

        ! Initialise the OpenCL system

        ! Create the kernel name
        if (knum .gt. 9) then
            write(kstr,'(a12,i2.2)') "mataccKernel", knum
        else
            write(kstr,'(a12,i1.1)') "mataccKernel", knum
        end if

        ! Can't pass a bare constant string as it has not been allocated        
        srcstr='matacc.cl'

        call oclInit(srcstr,kstr)
        call oclGetMaxComputeUnits(nunits)

#ifdef VERBOSE
        print *, "Number of compute units: ",nunits
#endif 

        ! Allocate space for results
        allocate( mC(nunits) )

        ! Create the buffers
        call oclMakeReadBuffer(mA_buf,4 * mSize) 
        call oclMakeWriteBuffer(mC_buf,4 * nunits) 
        ! setArg takes the index of the argument and a value of the same type as the kernel argument
        call oclSetFloatArrayArg(0, mA_buf )
        call oclSetFloatArrayArg(1, mC_buf)
        call oclSetIntConstArg(2, mWidth)

        do run=1,nruns
            call oclWriteBuffer(mA_buf,4 * mSize,mA)
        ! This is the actual "run" command.
            call dtime(tarray, result)
!   double tstart=wsecond()
            call runOcl(
#if KERNEL<7 || KERNEL==14
     &       nunits,
     &       1
#elif KERNEL<=8 || KERNEL==15
     &       nunits*16,
     &       16 ! 16 threads
#elif KERNEL==9 || KERNEL==16
     &       nunits*32,
     &       32 ! 32 threads
#elif KERNEL==10
     &       nunits*16,
     &       16 ! 16 threads
#elif KERNEL==12
     &       nunits*16,
     &       16 ! 16 threads
#elif KERNEL==13 || KERNEL==17
     &       nunits*64,
     &       64 ! 64 threads
#elif KERNEL==18 || KERNEL==20
     &       nunits*128,
     &       128 ! 128 threads
#elif KERNEL==19
     &       nunits*256,
     &       256 ! 256 threads
#elif KERNEL==11
     &       nunits,
     &       1 ! single thread
#endif
     &       )
            ! Read back the results
            call oclReadBuffer(mC_buf,4 * nunits,mC)
            call dtime(tarray, result)
#endif 
        !--------------------------------------------------------------------------------
        !----  Here ends the actual OpenCL part
        !--------------------------------------------------------------------------------
#ifdef VERBOSE
#if REF==1
            mCtot=0.0
            do i=1,nunits
                mCtot=mCtot+mC(i)
            end do
            correct=0               ! number of correct results returned
            if (mCtot .eq. mCref) then
                correct=correct+1
            end if
            print '(A2,F3.1,A2,F3.1)', "  ",mCtot,"<>",mCref
#endif
#if REF!=2
            print *, "OpenCL execution time: ",result*1000," ms"
        end do ! nruns
#endif
#else
#if REF!=2
            print *,  " ",result*1000," "
        end do ! nruns
#endif
#endif


      end
